"""Reconcile analytical and observed GPU memory for the Day 8 KV cache."""

import gc
from typing import cast

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

BLOCK_SIZE = 16
TARGET_GPU_UTILIZATION = 0.90
DTYPE = torch.float16

_BYTES_PER_KIB = 1024
_BYTES_PER_MIB = 1024**2
_BYTES_PER_GIB = 1024**3


def gib(num_bytes: int) -> float:
    """Convert bytes to GiB."""
    return num_bytes / _BYTES_PER_GIB


def mib(num_bytes: int) -> float:
    """Convert bytes to MiB."""
    return num_bytes / _BYTES_PER_MIB


def parameter_bytes(model: PreTrainedModel) -> int:
    """Return the total bytes occupied by model parameters."""
    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )


def main() -> None:
    """Measure model memory and derive prospective KV-cache capacity."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU memory reconciliation.")

    device = torch.device("cuda")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_before, total_memory = torch.cuda.mem_get_info()

    gpu_name = torch.cuda.get_device_name(device)

    print("=== Before model load ===")
    print(f"GPU: {gpu_name}")
    print(f"Total memory: {gib(total_memory):.2f} GiB")
    print(f"Free memory: {gib(free_before):.2f} GiB")
    print(f"PyTorch allocated: {gib(torch.cuda.memory_allocated()):.2f} GiB")
    print(f"PyTorch reserved: {gib(torch.cuda.memory_reserved()):.2f} GiB")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=DTYPE,
    )

    model_module = cast(torch.nn.Module, model)
    model_module.to(device)
    model_module.eval()

    torch.cuda.synchronize()

    free_after, _ = torch.cuda.mem_get_info()

    actual_parameter_bytes = parameter_bytes(model)
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    device_delta = free_before - free_after

    num_layers = int(model.config.num_hidden_layers)
    num_kv_heads = int(model.config.num_key_value_heads)

    head_dim_config = getattr(model.config, "head_dim", None)
    if head_dim_config is not None:
        head_dim = int(head_dim_config)
    else:
        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

    dtype_bytes = torch.empty((), dtype=DTYPE).element_size()

    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    kv_block_bytes = kv_bytes_per_token * BLOCK_SIZE

    target_used_bytes = int(total_memory * TARGET_GPU_UTILIZATION)
    currently_used_bytes = total_memory - free_after
    kv_budget_bytes = max(
        0,
        target_used_bytes - currently_used_bytes,
    )
    num_blocks = kv_budget_bytes // kv_block_bytes

    print()
    print("=== After model load ===")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"Dtype: {DTYPE}")
    print(f"Free memory: {gib(free_after):.2f} GiB")
    print(f"Device-memory delta: {gib(device_delta):.2f} GiB")
    print(
        f"Actual parameter bytes: "
        f"{actual_parameter_bytes:,} "
        f"({gib(actual_parameter_bytes):.4f} GiB)"
    )
    print(f"PyTorch allocated: {gib(allocated_bytes):.2f} GiB")
    print(f"PyTorch reserved: {gib(reserved_bytes):.2f} GiB")

    print()
    print("=== KV geometry ===")
    print(f"Layers: {num_layers}")
    print(f"KV heads: {num_kv_heads}")
    print(f"Head dimension: {head_dim}")
    print(f"Dtype bytes: {dtype_bytes}")
    print(f"Block size: {BLOCK_SIZE} tokens")
    print(
        f"KV bytes/token: {kv_bytes_per_token:,} "
        f"({kv_bytes_per_token / _BYTES_PER_KIB:.2f} KiB)"
    )
    print(f"KV bytes/block: {kv_block_bytes:,} ({mib(kv_block_bytes):.4f} MiB)")

    print()
    print("=== Prospective KV capacity ===")
    print(f"Target GPU utilization: {TARGET_GPU_UTILIZATION:.0%}")
    print(f"Target used memory: {gib(target_used_bytes):.2f} GiB")
    print(f"Currently used memory: {gib(currently_used_bytes):.2f} GiB")
    print(f"KV budget: {gib(kv_budget_bytes):.2f} GiB")
    print(f"Derived num_blocks: {num_blocks:,}")
    print(f"Derived token capacity: {num_blocks * BLOCK_SIZE:,}")


if __name__ == "__main__":
    main()
