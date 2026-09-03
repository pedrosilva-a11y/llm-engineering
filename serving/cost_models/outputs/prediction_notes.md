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
