# Paged KV and Packed Execution Validation

Validation results for the paged KV-cache implementation, covering physical storage
correctness, memory reconciliation against the analytical cost model, prompt-level
fragmentation, output equivalence against the frozen reference set, and packed
multi-sequence execution correctness on CUDA.

**Status:** all executed checks pass. The single-sequence paged path reproduces the frozen
reference exactly. The packed multi-sequence path matches an independent naive runner
across prefill, decode, and mixed execution on CUDA and reproduces all five frozen Day 7
reference cases exactly.

| Environment | |
|---|---|
| GPU | Tesla T4 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Validation dtypes | float16, float32 |
| Transformers | 5.17.0 |

---

## 1. Single-sequence output equivalence

This section records the Day 8 single-sequence paged gate. The paged implementation
must produce the same tokens as the contiguous reference generated before any paging
existed. Section 5 repeats the frozen-reference gate through concurrent packed execution.

The frozen Day 7 reference set was generated in float16. Its attention backend was not
explicitly pinned during reference generation, which the token-level equality contract
below accommodates.

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

Backend and dtype behavior observed during packed validation is documented in
Section 5, including an FP16 eager-attention NaN finding on the T4 environment.

---

## 2. Physical storage validation

Direct validation of the KV storage layer on CUDA, independent of any model.

```text
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

```text
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

```text
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

## 5. Packed multi-sequence execution

Day 9a replaced sequential per-sequence Hugging Face forwards with one packed model
forward. Correctness was validated against `NaiveModelRunner`, which recomputes each
complete sequence history independently with KV caching disabled.

`NaiveModelRunner` ignores execution phase and always recomputes the complete token
history, so decode comparisons remain independent of the paged KV state.

Both runners shared the exact same loaded model object so that checkpoint, weights,
dtype, device, and attention backend were held constant. Fresh paged KV storage was
created for each validation case.

### FP32 eager differential

The first CUDA differential used float32 with eager attention.

| Case | Max absolute logit diff | Smallest top-2 margin | Top-1 |
|---|---:|---:|---|
| unequal prefill + prefill | 0.00002635 | 1.42997 | PASS |
| unequal decode + decode | 0.00002146 | 0.06114 | PASS |
| decode + prefill | 0.00001359 | 0.08971 | PASS |

Each row reports the maximum absolute logit difference observed across sequences in that
case and the smallest top-2 margin observed across both runners and all sequences in that
case.

All predictions matched. The small numerical differences are consistent with
floating-point operation-order differences; no evidence of a semantic difference
between the independently recomputed and packed execution paths was observed.

The decode cases are particularly important because the paged runner first populated
the cache with real model-produced prefix KV states, then processed only the newest
token while gathering the complete history from paged storage. The naive runner
independently recomputed the same full history from token ids.

### FP16 backend finding

An initial float16 eager-attention run produced only NaN logits. The failure was
isolated outside the serving engine with a minimal Hugging Face forward using the same
Qwen checkpoint:

- float16 + eager attention: all logits NaN
- float16 + SDPA: all logits finite
- float32 + eager attention: finite during the runner differential

The exact operation that first produces the NaN under float16 eager attention was not
localized, so this is recorded as an environment/backend finding rather than attributed
to a particular kernel or model operation.

### FP16 SDPA differential

Repeating the packed differential with float16 and SDPA produced finite logits and
matching top-1 predictions in all cases:

| Case | Max absolute logit diff | Smallest top-2 margin | Top-1 |
|---|---:|---:|---|
| unequal prefill + prefill | 0.02734375 | 1.42187500 | PASS |
| unequal decode + decode | 0.02050781 | 0.05468750 | PASS |
| decode + prefill | 0.01171875 | 0.08593750 | PASS |

Each row reports the maximum absolute logit difference observed across sequences in that
case and the smallest top-2 margin observed across both runners and all sequences in that
case.

This also establishes that the explicit 4D additive block-diagonal attention mask used
by the packed runner is accepted and behaves correctly through the SDPA path for these
validation cases.

The correctness criterion remains token-level/top-1 equality rather than bitwise logit
equality. FP16 differences are materially larger than the FP32 differences, as
expected, but no tested case changed the selected token.

### Frozen Day 7 reference gate

The final packed correctness gate used the original five-case Day 7 frozen reference
set, generated before paged KV caching and packed execution existed. All five requests
were submitted concurrently so the runner exercised multi-sequence packed prefill and
decode rather than single-sequence forwarding.

The validation used the same model revision, float16 dtype, Tesla T4 GPU, and
Transformers 5.17.0 environment recorded by the frozen artifact. SDPA was explicitly
selected for the packed run; the frozen artifact did not pin its attention backend, so
backend equality with the original generation is not asserted.

| Case | Prompt | Generated | Finish | Result |
|---|---:|---:|---|---|
| `natural_eos` | 37 | 20 | eos | PASS |
| `length_termination` | 44 | 4 | length | PASS |
| `exact_block_boundary` | 48 | 8 | length | PASS |
| `decode_crosses_block_boundary` | 47 | 8 | length | PASS |
| `long_multi_block_prompt` | 55 | 12 | length | PASS |

**All five matched exactly** — identical generated token identifiers and identical
finish reasons through packed multi-sequence execution.

---

## 6. What this does not establish

- **Production-efficient packed attention.** Multi-sequence execution is validated for
  correctness, but the current block-diagonal mask is dense. It establishes semantic
  isolation, not sparse/paged-kernel FLOP efficiency.
- **Performance.** No timing was collected. Gather and scatter are implemented as
  ordinary indexing operations, not fused kernels, and are expected to be substantially
  slower than a specialized implementation.
- **Aggregate fragmentation under concurrency.** The figures above are per-prompt and
  single-sequence.
- **Sampling paths.** Reference validation uses greedy decoding, since sampling depends
  on generator state and call ordering.
- **Sustained memory pressure and packed re-prefill.** The packed frozen-reference gate
  ran without preemption. Preemption followed by re-prefill through packed execution
  therefore remains unvalidated on device.

---

## 7. Reproducing

```text
uv run python -m scripts.validate_paged_kv_cuda       # storage correctness
uv run python -m scripts.reconcile_gpu_memory         # memory reconciliation
uv run python -m scripts.validate_reference_outputs   # packed frozen-reference gate
uv run python -m scripts.measure_kv_fragmentation     # fragmentation (CPU)
uv run python -m scripts.validation.validate_packed_runner_cuda  # packed CUDA differential
```

The packed differential lives under `scripts/validation/`; the earlier Day 7 and Day 8
validation utilities remain top-level modules under `scripts/`.

All validation commands above except fragmentation require CUDA and are excluded from
the default test target, which remains GPU-free. Reference data lives in
`reference_outputs/day7_qwen2_5_1_5b.json`, with full generation conditions recorded
alongside the outputs.
