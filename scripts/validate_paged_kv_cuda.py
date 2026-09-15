"""Validate paged KV-cache storage operations on CUDA."""

import torch

from serving.engine.paged_kv_storage import PagedKVStorage

NUM_LAYERS = 28
NUM_BLOCKS = 256
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
DTYPE = torch.float16


def main() -> None:
    """Validate CUDA allocation and paged KV scatter/gather behavior."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for paged KV-cache validation.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    storage = PagedKVStorage(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=device,
    )

    layer_index = 7

    # Deliberately non-contiguous physical slots.
    slot_ids = (0, 17, 255, 1024, 4095)

    num_tokens = len(slot_ids)

    keys = torch.arange(
        num_tokens * NUM_KV_HEADS * HEAD_DIM,
        dtype=DTYPE,
        device=device,
    ).reshape(
        num_tokens,
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    values = keys + 100.0

    storage.scatter(
        layer_index=layer_index,
        slot_ids=slot_ids,
        keys=keys,
        values=values,
    )

    gathered_keys, gathered_values = storage.gather(
        layer_index=layer_index,
        slot_ids=slot_ids,
    )

    if not torch.equal(gathered_keys, keys):
        raise AssertionError("Gathered keys do not exactly match scattered keys.")

    if not torch.equal(gathered_values, values):
        raise AssertionError("Gathered values do not exactly match scattered values.")

    expected_shape = (
        NUM_LAYERS,
        NUM_BLOCKS * BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    if storage.key_cache.shape != expected_shape:
        raise AssertionError(f"Unexpected key-cache shape: {storage.key_cache.shape}.")

    if storage.value_cache.shape != expected_shape:
        raise AssertionError(
            f"Unexpected value-cache shape: {storage.value_cache.shape}."
        )

    if storage.key_cache.dtype != DTYPE:
        raise AssertionError(f"Unexpected key-cache dtype: {storage.key_cache.dtype}.")

    if storage.value_cache.dtype != DTYPE:
        raise AssertionError(
            f"Unexpected value-cache dtype: {storage.value_cache.dtype}."
        )

    torch.cuda.synchronize()

    cache_bytes = (
        storage.key_cache.numel() * storage.key_cache.element_size()
        + storage.value_cache.numel() * storage.value_cache.element_size()
    )

    print("=== CUDA paged KV validation ===")
    print(f"GPU: {gpu_name}")
    print(f"Dtype: {DTYPE}")
    print(f"Layers: {NUM_LAYERS}")
    print(f"Blocks: {NUM_BLOCKS}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Physical slots: {storage.num_slots}")
    print(f"KV heads: {NUM_KV_HEADS}")
    print(f"Head dimension: {HEAD_DIM}")
    print(f"Cache shape: {expected_shape}")
    print(f"Allocated KV bytes: {cache_bytes:,}")
    print(f"Allocated KV MiB: {cache_bytes / 1024**2:.2f}")
    print(f"Validated physical slots: {slot_ids}")

    print()
    print("CUDA allocation: PASS")
    print("FP16 storage: PASS")
    print("Non-contiguous scatter: PASS")
    print("Logical-order gather: PASS")
    print("Exact round trip: PASS")


if __name__ == "__main__":
    main()
