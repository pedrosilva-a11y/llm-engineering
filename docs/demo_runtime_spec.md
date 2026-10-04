# Demo Runtime Specification

**Status:** Target runtime configuration for the inference-engine demo  
**Purpose:** Freeze the model and accelerator assumptions before frontend/backend integration and final GPU deployment.

## 1. Document placement

Recommended repository path:

```text
docs/demo_runtime_spec.md
```

This document belongs under `docs/` because it describes the reproducible runtime target for the demo rather than implementation details owned by `serving/engine/`.

## 2. Accelerator

| Field | Target |
|---|---|
| Accelerator | NVIDIA L4 |
| VRAM | 24 GB |
| Execution device | CUDA |
| Precision | FP16 |
| Attention backend | SDPA |
| Cloud provider | To be selected |
| Instance lifetime | Demo development, benchmark, rehearsal, and presentation |

The L4 is the target accelerator for the final demo and benchmark environment.

The provider is intentionally left open so platform choice can be made independently from the engine configuration.

## 3. Model

| Field | Target |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Model type | Decoder-only causal language model |
| Precision | FP16 |
| Attention backend | SDPA |
| EOS token ID | `151645` |

The model revision is pinned so the demo, benchmark results, and existing validation artifacts refer to the same checkpoint.

## 4. Engine configuration

The final GPU-backed demo should use the existing packed paged execution path:

```text
Engine
→ Scheduler
→ ModelExecution batch
→ PagedModelRunner
→ BatchLayout / PackedInputs
→ HFBatchedPagedCache
→ PagedKVCache
→ PagedKVStorage
→ Qwen on CUDA
```

Baseline KV configuration:

| Field | Value |
|---|---:|
| Block size | 16 tokens |
| Demo block count | 256 |
| Physical KV slots | 4,096 |
| KV bytes per token | 28,672 bytes / 28 KiB |

The 256-block configuration is a demo baseline, not the maximum capacity of an L4. Final benchmark capacity may be derived from measured free GPU memory after model load.

## 5. Demo architecture

During local development:

```text
Local frontend
    ↓
Local FastAPI gateway
    ↓
Development/stub engine
```

During the final GPU demo:

```text
Local frontend
    ↓ HTTP / SSE
Remote FastAPI gateway on GPU instance
    ↓
Real packed paged inference engine
    ↓
Qwen2.5-1.5B-Instruct
    ↓
NVIDIA L4
```

The frontend should depend only on the HTTP/SSE contract. Switching from local development to the GPU-backed demo should require changing the backend base URL rather than frontend behavior.

## 6. Runtime metadata shown in the frontend

The frontend may display a compact static runtime line such as:

```text
NVIDIA L4 · Qwen2.5-1.5B-Instruct · FP16 · SDPA · 256 × 16 = 4,096 KV slots
```

This information should be fetched once at startup or supplied as deployment configuration rather than polled continuously.

A single lightweight dynamic allocator metric such as free KV blocks may be added later if useful for the demo.

## 7. Acceptance criteria for the GPU environment

Before the GPU environment becomes the final demo target, it must satisfy all of the following:

- NVIDIA L4 is visible through CUDA.
- Python 3.11 project environment installs successfully with `uv`.
- The pinned Qwen tokenizer integration test passes:

  ```bash
  RUN_TOKENIZER_INTEGRATION=1 uv run pytest \
    serving/gateway/tests/test_tokenizer_integration.py -v
  ```

- The pinned Qwen revision loads successfully in FP16.
- SDPA produces finite logits.
- The real `PagedModelRunner` completes a request successfully.
- The FastAPI gateway can stream a real completion end to end.
- Multiple concurrent requests exercise packed multi-sequence execution.
- Existing quality checks remain green.

## 8. Immediate development sequence

GPU rental is not required for the next local steps.

Proceed in this order:

```text
1. Freeze this runtime specification.
2. Run the existing gateway locally.
3. Verify the local API/SSE contract with curl.
4. Add or finalize the text-facing gateway boundary.
5. Build the frontend against the local backend.
6. Make the backend URL configurable.
7. Provision the L4.
8. Deploy the real engine and model to the GPU instance.
9. Point the unchanged local frontend at the remote backend.
10. Validate concurrency and run final benchmarks.
```

The goal is to complete as much API and frontend work as possible locally before paying for GPU runtime, while still moving to the L4 early enough to leave time for real integration and benchmark debugging.
