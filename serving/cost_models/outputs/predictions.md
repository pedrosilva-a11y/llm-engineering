# LLM Serving Cost-Model Predictions

## Configuration

- Model: qwen2.5-1.5b
- Hardware: L4
- Bytes per value: 2
- Model parameters: 1,543,656,960
- Model weight memory: 2.88 GiB
- KV cache per token: 28.00 KiB
- Hardware ridge point: 403.33 FLOPs/byte

## Memory Estimates

| Context Length | KV cache / sequence | Max concurrency |
| ---: | ---: | ---: |
| 512 | 14.00 MiB | 1,435 |
| 1,024 | 28.00 MiB | 717 |
| 2,048 | 56.00 MiB | 358 |
| 4,096 | 112.00 MiB | 179 |
| 8,192 | 224.00 MiB | 89 |

## Decode Estimates

| Context Length | Batch Size | FLOPs / token | Arithmetic Intensity | Ideal TPOT | Ideal Throughput | Ridge Point | Regime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 512 | 1 | 2,708,471,808 | 0.87 FLOPs/byte | 10.34 ms | 96.71 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 512 | 8 | 2,708,471,808 | 6.76 FLOPs/byte | 10.68 ms | 748.83 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 512 | 32 | 2,708,471,808 | 24.36 FLOPs/byte | 11.86 ms | 2,698.15 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 512 | 128 | 2,708,471,808 | 69.75 FLOPs/byte | 16.57 ms | 7,726.31 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 1,024 | 1 | 2,796,552,192 | 0.90 FLOPs/byte | 10.39 ms | 96.26 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 1,024 | 8 | 2,796,552,192 | 6.73 FLOPs/byte | 11.07 ms | 722.36 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 1,024 | 32 | 2,796,552,192 | 22.22 FLOPs/byte | 13.43 ms | 2,383.46 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 1,024 | 128 | 2,796,552,192 | 52.26 FLOPs/byte | 22.83 ms | 5,606.59 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 2,048 | 1 | 2,972,712,960 | 0.94 FLOPs/byte | 10.49 ms | 95.36 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 2,048 | 8 | 2,972,712,960 | 6.69 FLOPs/byte | 11.86 ms | 674.67 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 2,048 | 32 | 2,972,712,960 | 19.15 FLOPs/byte | 16.56 ms | 1,932.65 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 2,048 | 128 | 2,972,712,960 | 35.87 FLOPs/byte | 35.36 ms | 3,620.19 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 4,096 | 1 | 3,325,034,496 | 1.04 FLOPs/byte | 10.68 ms | 93.61 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 4,096 | 8 | 3,325,034,496 | 6.61 FLOPs/byte | 13.42 ms | 595.97 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 4,096 | 32 | 3,325,034,496 | 15.54 FLOPs/byte | 22.82 ms | 1,402.21 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 4,096 | 128 | 3,325,034,496 | 23.48 FLOPs/byte | 60.41 ms | 2,118.81 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 8,192 | 1 | 4,029,677,568 | 1.21 FLOPs/byte | 11.07 ms | 90.30 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 8,192 | 8 | 4,029,677,568 | 6.49 FLOPs/byte | 16.56 ms | 483.23 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 8,192 | 32 | 4,029,677,568 | 12.16 FLOPs/byte | 35.35 ms | 905.28 tokens/s | 403.33 FLOPs/byte | Memory-bound |
| 8,192 | 128 | 4,029,677,568 | 15.56 FLOPs/byte | 110.52 ms | 1,158.17 tokens/s | 403.33 FLOPs/byte | Memory-bound |

## Prefill Estimates

| Prompt Length | Batch Size | Arithmetic Intensity | Ideal Prefill Time | Ridge Point | Regime |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 128 | 1 | 108.97 FLOPs/byte | 10.30 ms | 403.33 FLOPs/byte | Memory-bound |
| 256 | 1 | 218.60 FLOPs/byte | 10.32 ms | 403.33 FLOPs/byte | Memory-bound |
| 384 | 1 | 328.87 FLOPs/byte | 10.33 ms | 403.33 FLOPs/byte | Memory-bound |
| 512 | 1 | 439.79 FLOPs/byte | 11.27 ms | 403.33 FLOPs/byte | Compute-bound |
| 1,024 | 1 | 889.91 FLOPs/byte | 22.92 ms | 403.33 FLOPs/byte | Compute-bound |
| 2,048 | 1 | 1,820.55 FLOPs/byte | 47.33 ms | 403.33 FLOPs/byte | Compute-bound |


## Pre-measurement Predictions

Written September 3, 2026, before running any GPU measurements.

These predictions target Qwen2.5-1.5B in BF16 on an NVIDIA L4. The analytical values in `analytical_report.md` are ideal roofline estimates and intentionally exclude framework overhead, kernel inefficiency, scheduling, synchronization, and other runtime effects.

### 1. Short-prompt prefill will be nearly flat

For isolated, warm requests, I predict that measured prefill latency for 128-token and 384-token prompts will be within 15% of each other.

The analytical model predicts approximately 10.30 ms at 128 tokens and 10.33 ms at 384 tokens because both workloads remain memory-bound and are dominated by reading approximately 2.88 GiB of model weights.

I expect this prediction to fail if runtime overhead or kernel behavior introduces stronger sequence-length dependence than the roofline model captures.

### 2. TTFT at 512 tokens

The ideal roofline prefill estimate for a 512-token prompt is approximately 11.27 ms.

For a warm, isolated request without queueing or network latency, I predict measured TTFT will be approximately 15–25 ms.

I expect measured TTFT to exceed the analytical floor because the model assumes ideal hardware utilization and excludes runtime and sampling overhead.

### 3. Batch-1 decode TPOT

At a context length of 2,048 tokens, the analytical batch-1 decode-step floor is approximately 10.49 ms.

I predict measured batch-1 TPOT will be approximately 15–22 ms.

Decode should remain strongly memory-bandwidth-bound at this workload.

### 4. Decode throughput knee

At a 2,048-token context, I predict that decode throughput will begin showing strong diminishing returns around concurrency 128.

The analytical model predicts approximately:

- Batch 32: 1,933 tokens/s
- Batch 128: 3,620 tokens/s
- Batch 256: 4,237 tokens/s
- Theoretical memory ceiling of 358 sequences: 4,453 tokens/s

I therefore expect increasing concurrency beyond roughly 128–256 sequences to produce substantially smaller throughput improvements while continuing to increase per-request TPOT.

### 5. Memory-bandwidth utilization

For memory-bound decode workloads, I predict sustained effective memory bandwidth will reach approximately 55–70% of the L4's theoretical 300 GB/s peak.

This corresponds to approximately 165–210 GB/s of achieved bandwidth.

### Where I expect the model to be most wrong

I expect the largest errors in absolute latency predictions.

The roofline model assumes ideal use of either compute throughput or memory bandwidth and omits kernel-launch overhead, framework/runtime costs, activation traffic, synchronization, cache behavior, scheduling, and implementation-specific kernel efficiency.

I expect the model to be more reliable at predicting qualitative behavior—memory-bound versus compute-bound regimes, KV-cache scaling, batching benefits, and diminishing throughput returns—than exact TTFT or TPOT values.
