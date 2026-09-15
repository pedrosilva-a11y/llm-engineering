# Paged KV Cache Validation

Validation results for the paged KV-cache implementation, covering physical storage
correctness, memory reconciliation against the analytical cost model, prompt-level
fragmentation, and output equivalence against the frozen reference set.

**Status:** all checks pass. The paged path reproduces the contiguous reference exactly.

| Environment | |
|---|---|
| GPU | Tesla T4 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Dtype | float16 |
| Transformers | 5.17.0 |

---

## 1. Output equivalence

The gate. The paged implementation must produce the same tokens as the contiguous
reference generated before any paging existed.

| Case | Prompt | Generated | Finish | Result |
|---|---:|---:|---|---|
| `natural_eos` | 37 | 20 | eos | PASS |
| `length_termination` | 44 | 4 | length | PASS |
| `exact_block_boundary` | 48 | 8 | length | PASS |
| `decode_crosses_block_boundary` | 47 | 8 | length | PASS |
| `long_multi_block_prompt` | 55 | 12 | length | PASS |

**All five matched exactly** — identical generated token identifiers and identical finish
reasons. No divergence required diagnosis.

Two cases carry disproportionate weight. `exact_block_boundary` uses a 48-token prompt,
exactly three full blocks, so the final prompt slot sits on a block edge.
`decode_crosses_block_boundary` uses 47 tokens, leaving one free slot, so the very first
generated token forces a new block allocation. Between them they exercise the
off-by-one surface in slot arithmetic during both prefill and the first decode step.

**What equivalence was required to be.** Token-level equality, not bitwise logit
equality. Framework attention selects kernels by tensor shape, and the paged path can
present different shapes than a contiguous cache, which may select a different backend
with different floating-point reduction order. In practice no divergence occurred at
all, but the criterion was chosen in advance to avoid mistaking numerics for a bug.

---

## 2. Physical storage validation

Direct validation of the KV storage layer on CUDA, independent of any model.

```
Cache shape: (28, 4096, 2, 128)     layers × slots × kv_heads × head_dim
Blocks: 256 × block size 16 = 4,096 physical slots
Allocated: 112.00 MiB
```

| Check | Result |
|---|---|
| CUDA allocation | PASS |
| FP16 storage | PASS |
| Non-contiguous scatter | PASS |
| Logical-order gather | PASS |
| Exact round trip | PASS |

Validated slots `(0, 17, 255, 1024, 4095)` — deliberately scattered, including both
extremes of the address space and a mid-block offset.

**The non-contiguous check is the one that matters.** It writes to physically scattered
slots and reads back in logical token order, confirming the block table's indirection
works rather than the implementation quietly relying on contiguous allocation. A
contiguous implementation passes every other check in this table.

Note the validation configuration uses 256 blocks rather than the derived capacity from
section 3. Correctness does not depend on pool size, and a small pool keeps the check
fast.

---

## 3. Memory reconciliation

Comparing the analytical cost model, written before any GPU access, against measured
device memory.

### Before and after model load

| | GiB |
|---|---:|
| Total device memory | 14.56 |
| Free before load | 14.46 |
| Free after load | 11.40 |
| Device delta | 3.06 |
| PyTorch allocated | 2.88 |
| PyTorch reserved | 3.06 |

The 0.10 GiB used before loading is the CUDA context. The gap between allocated (2.88)
and reserved (3.06) is PyTorch's caching allocator holding memory it has requested from
the driver but not handed to tensors.

### Predicted versus measured parameter bytes

| | Bytes |
|---|---:|
| Cost-model prediction | 3,087,313,920 |
| Measured | 3,087,428,608 |
| Difference | 114,688 |
| Relative error | **0.0037%** |

**The entire discrepancy is accounted for.** `ModelSpecification` documents that it
excludes attention and MLP projection biases, and names Qwen2.5 as an architecture that
carries them. Qwen2.5 applies biases to the query, key and value projections but not the
output projection:

```
per layer:  q_dim + 2 × kv_dim  =  1536 + 512  =  2,048
× 28 layers                                    =  57,344 parameters
× 2 bytes (fp16)                               = 114,688 bytes
```

That is exactly the measured difference, to the byte. The analytical model was correct,
and its single documented omission explains the whole of the remaining error.

### KV geometry

| | Value |
|---|---:|
| Layers | 28 |
| KV heads | 2 |
| Head dimension | 128 |
| Bytes per element | 2 |
| Block size | 16 tokens |
| **KV bytes per token** | **28,672** (28.00 KiB) |
| KV bytes per block | 458,752 (0.4375 MiB) |

Both figures match the cost model exactly. Note the use of `n_kv_head = 2` rather than
`n_head = 12`: grouped-query attention makes the KV cache six times smaller than
multi-head attention would, and this is where that saving becomes physical.

### Derived capacity

| | Value |
|---|---:|
| Target utilization | 90% |
| Target used memory | 13.11 GiB |
| Currently used | 3.16 GiB |
| KV budget | 9.94 GiB |
| Derived blocks | 23,274 |
| Token capacity | 372,384 |

At 2,048 tokens of context that is approximately **181 concurrent sequences**.

The Day 1 predictions assumed an L4 with 22.5 GiB, giving 19.62 GiB of KV budget and
358 concurrent sequences. The T4 result is close to half, tracking the memory ratio as
expected. **Predictions written for an L4 must be reconciled against L4 measurements**;
the benchmark phase should pin that card rather than accept whichever GPU is allocated.

### A note on the budgeting formula

`torch.cuda.mem_get_info()` returns *free* memory, which already reflects loaded
weights. Subtracting weight bytes from it again would double-count them. The correct
form is:

```
target_used    = total × utilization
currently_used = total − free_after_load
kv_budget      = max(0, target_used − currently_used)
```

---

## 4. Prompt fragmentation

Internal fragmentation measured at admission, when blocks are allocated for the prompt.
Measuring after completion would report zero, since the scheduler releases a request's
blocks when it finishes.

| Case | Tokens | Blocks | Slots | Wasted | Utilization |
|---|---:|---:|---:|---:|---:|
| `natural_eos` | 37 | 3 | 48 | 11 | 77.08% |
| `length_termination` | 44 | 3 | 48 | 4 | 91.67% |
| `exact_block_boundary` | 48 | 3 | 48 | 0 | 100.00% |
| `decode_crosses_block_boundary` | 47 | 3 | 48 | 1 | 97.92% |
| `long_multi_block_prompt` | 55 | 4 | 64 | 9 | 85.94% |

**Aggregate:** 231 prompt tokens across 256 allocated slots — 25 wasted, **90.23%
utilization**.

Internal fragmentation is bounded by `block_size − 1` slots per sequence, so worst-case
waste for 16-token blocks is 15 slots regardless of sequence length. The observed spread
from 0% to 23% waste is a function of where each prompt length falls relative to a block
boundary, and it shrinks in relative terms as sequences grow.

**What this baseline is for.** Prefix caching reduces *duplication* between sequences
sharing a prompt, not internal fragmentation within a block. Recording both separately
means the two effects can be attributed independently once caching lands.

---

## 5. What this does not establish

- **Batched execution.** The cache adapter is documented and enforced as single-sequence.
  Multi-sequence forward passes are not yet validated.
- **Performance.** No timing was collected. Gather and scatter are implemented as
  ordinary indexing operations, not fused kernels, and are expected to be substantially
  slower than a specialized implementation.
- **Aggregate fragmentation under concurrency.** The figures above are per-prompt and
  single-sequence.
- **Sampling paths.** Reference validation uses greedy decoding, since sampling depends
  on generator state and call ordering.
- **Sustained memory pressure.** Preemption and re-prefill interact with paged storage
  but were not exercised on device.

---

## 6. Reproducing

```
uv run python -m scripts.validate_paged_kv_cuda       # storage correctness
uv run python -m scripts.reconcile_gpu_memory         # memory reconciliation
uv run python -m scripts.validate_reference_outputs   # equivalence gate
uv run python -m scripts.measure_kv_fragmentation     # fragmentation (CPU)
```

The first three require CUDA and are excluded from the default test target, which
remains GPU-free. Reference data lives in
`reference_outputs/day7_qwen2_5_1_5b.json`, with full generation conditions recorded
alongside the outputs.
