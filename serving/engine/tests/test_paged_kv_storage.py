"""Tests for flat paged KV-cache storage."""

import pytest
import torch

from serving.engine.paged_kv_storage import PagedKVStorage


def test_storage_allocates_expected_cache_shape() -> None:
    """Allocate key and value caches with the expected physical geometry."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert storage.num_slots == 12
    assert storage.key_cache.shape == (2, 12, 2, 8)
    assert storage.value_cache.shape == (2, 12, 2, 8)


def test_storage_uses_configured_dtype() -> None:
    """Allocate key and value caches with the configured dtype."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert storage.key_cache.dtype == torch.float16
    assert storage.value_cache.dtype == torch.float16


def test_storage_uses_configured_device() -> None:
    """Allocate key and value caches on the configured device."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert storage.key_cache.device == torch.device("cpu")
    assert storage.value_cache.device == torch.device("cpu")


@pytest.mark.parametrize(
    (
        "num_layers",
        "num_blocks",
        "block_size",
        "num_kv_heads",
        "head_dim",
        "field_name",
    ),
    [
        (0, 3, 4, 2, 8, "num_layers"),
        (-1, 3, 4, 2, 8, "num_layers"),
        (2, 0, 4, 2, 8, "num_blocks"),
        (2, -1, 4, 2, 8, "num_blocks"),
        (2, 3, 0, 2, 8, "block_size"),
        (2, 3, -1, 2, 8, "block_size"),
        (2, 3, 4, 0, 8, "num_kv_heads"),
        (2, 3, 4, -1, 8, "num_kv_heads"),
        (2, 3, 4, 2, 0, "head_dim"),
        (2, 3, 4, 2, -1, "head_dim"),
    ],
)
def test_storage_rejects_non_positive_dimensions(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    field_name: str,
) -> None:
    """Reject non-positive KV-cache dimensions."""
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be greater than zero",
    ):
        PagedKVStorage(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


# Scatter


def test_scatter_writes_keys_and_values_to_physical_slots() -> None:
    """Write key and value tensors into the requested physical slots."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    slot_ids = (1, 6, 10)

    keys = torch.arange(
        3 * 2 * 8,
        dtype=torch.float32,
    ).reshape(3, 2, 8)

    values = keys + 100.0

    storage.scatter(
        layer_index=0,
        slot_ids=slot_ids,
        keys=keys,
        values=values,
    )

    for logical_index, slot_id in enumerate(slot_ids):
        torch.testing.assert_close(
            storage.key_cache[0, slot_id],
            keys[logical_index],
        )
        torch.testing.assert_close(
            storage.value_cache[0, slot_id],
            values[logical_index],
        )


def test_scatter_supports_non_contiguous_physical_slots() -> None:
    """Write logical token order into non-contiguous physical slots."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    slot_ids = (0, 7, 12)

    keys = torch.stack(
        (
            torch.full((2, 8), 1.0),
            torch.full((2, 8), 2.0),
            torch.full((2, 8), 3.0),
        )
    )
    values = keys + 10.0

    storage.scatter(
        layer_index=0,
        slot_ids=slot_ids,
        keys=keys,
        values=values,
    )

    torch.testing.assert_close(storage.key_cache[0, 0], keys[0])
    torch.testing.assert_close(storage.key_cache[0, 7], keys[1])
    torch.testing.assert_close(storage.key_cache[0, 12], keys[2])

    torch.testing.assert_close(storage.value_cache[0, 0], values[0])
    torch.testing.assert_close(storage.value_cache[0, 7], values[1])
    torch.testing.assert_close(storage.value_cache[0, 12], values[2])


def test_scatter_writes_only_selected_layer() -> None:
    """Leave other transformer layers unchanged during a scatter."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    storage.key_cache.zero_()
    storage.value_cache.zero_()

    keys = torch.ones((2, 2, 8))
    values = torch.full((2, 2, 8), 2.0)

    storage.scatter(
        layer_index=1,
        slot_ids=(1, 5),
        keys=keys,
        values=values,
    )

    assert torch.count_nonzero(storage.key_cache[0]) == 0
    assert torch.count_nonzero(storage.value_cache[0]) == 0

    torch.testing.assert_close(storage.key_cache[1, 1], keys[0])
    torch.testing.assert_close(storage.key_cache[1, 5], keys[1])
    torch.testing.assert_close(storage.value_cache[1, 1], values[0])
    torch.testing.assert_close(storage.value_cache[1, 5], values[1])


@pytest.mark.parametrize("layer_index", [-1, 2])
def test_scatter_rejects_invalid_layer_index(layer_index: int) -> None:
    """Reject writes to layers outside the configured range."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((1, 2, 8))
    values = torch.ones((1, 2, 8))

    with pytest.raises(
        ValueError,
        match="layer_index must be within the configured layers",
    ):
        storage.scatter(
            layer_index=layer_index,
            slot_ids=(0,),
            keys=keys,
            values=values,
        )


@pytest.mark.parametrize("slot_id", [-1, 8])
def test_scatter_rejects_invalid_physical_slot(slot_id: int) -> None:
    """Reject writes outside the physical slot address space."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((1, 2, 8))
    values = torch.ones((1, 2, 8))

    with pytest.raises(
        ValueError,
        match="slot_ids must be within the configured physical slots",
    ):
        storage.scatter(
            layer_index=0,
            slot_ids=(slot_id,),
            keys=keys,
            values=values,
        )


def test_scatter_rejects_incorrect_key_shape() -> None:
    """Reject key tensors that do not match the storage geometry."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((2, 2, 7))
    values = torch.ones((2, 2, 8))

    with pytest.raises(ValueError, match="keys must have shape"):
        storage.scatter(
            layer_index=0,
            slot_ids=(0, 1),
            keys=keys,
            values=values,
        )


def test_scatter_rejects_incorrect_value_shape() -> None:
    """Reject value tensors that do not match the storage geometry."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((2, 2, 8))
    values = torch.ones((2, 2, 7))

    with pytest.raises(ValueError, match="values must have shape"):
        storage.scatter(
            layer_index=0,
            slot_ids=(0, 1),
            keys=keys,
            values=values,
        )


def test_scatter_rejects_incorrect_dtype() -> None:
    """Reject key/value tensors whose dtype differs from the cache."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((1, 2, 8), dtype=torch.float16)
    values = torch.ones((1, 2, 8), dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match="keys dtype must match the KV-cache dtype",
    ):
        storage.scatter(
            layer_index=0,
            slot_ids=(0,),
            keys=keys,
            values=values,
        )


# Gather


def test_gather_reads_keys_and_values_in_slot_order() -> None:
    """Read physical KV slots in the requested logical order."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    storage.key_cache.zero_()
    storage.value_cache.zero_()

    slot_ids = (1, 6, 10)

    expected_keys = torch.arange(
        3 * 2 * 8,
        dtype=torch.float32,
    ).reshape(3, 2, 8)

    expected_values = expected_keys + 100.0

    for logical_index, slot_id in enumerate(slot_ids):
        storage.key_cache[0, slot_id].copy_(expected_keys[logical_index])
        storage.value_cache[0, slot_id].copy_(expected_values[logical_index])

    keys, values = storage.gather(
        layer_index=0,
        slot_ids=slot_ids,
    )

    torch.testing.assert_close(keys, expected_keys)
    torch.testing.assert_close(values, expected_values)


def test_gather_preserves_logical_order_for_non_contiguous_slots() -> None:
    """Preserve requested logical order across non-contiguous physical slots."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    storage.key_cache.zero_()
    storage.value_cache.zero_()

    storage.key_cache[0, 12].fill_(1.0)
    storage.key_cache[0, 3].fill_(2.0)
    storage.key_cache[0, 9].fill_(3.0)

    storage.value_cache[0, 12].fill_(11.0)
    storage.value_cache[0, 3].fill_(12.0)
    storage.value_cache[0, 9].fill_(13.0)

    keys, values = storage.gather(
        layer_index=0,
        slot_ids=(12, 3, 9),
    )

    assert torch.all(keys[0] == 1.0)
    assert torch.all(keys[1] == 2.0)
    assert torch.all(keys[2] == 3.0)

    assert torch.all(values[0] == 11.0)
    assert torch.all(values[1] == 12.0)
    assert torch.all(values[2] == 13.0)


def test_scatter_then_gather_round_trip() -> None:
    """Recover the original KV tensors after scattering and gathering."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    slot_ids = (2, 7, 13)

    keys = torch.arange(
        3 * 2 * 8,
        dtype=torch.float32,
    ).reshape(3, 2, 8)

    values = keys + 1000.0

    storage.scatter(
        layer_index=1,
        slot_ids=slot_ids,
        keys=keys,
        values=values,
    )

    gathered_keys, gathered_values = storage.gather(
        layer_index=1,
        slot_ids=slot_ids,
    )

    torch.testing.assert_close(gathered_keys, keys)
    torch.testing.assert_close(gathered_values, values)


@pytest.mark.parametrize("layer_index", [-1, 2])
def test_gather_rejects_invalid_layer_index(layer_index: int) -> None:
    """Reject reads from layers outside the configured range."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        ValueError,
        match="layer_index must be within the configured layers",
    ):
        storage.gather(
            layer_index=layer_index,
            slot_ids=(0,),
        )


@pytest.mark.parametrize("slot_id", [-1, 8])
def test_gather_rejects_invalid_physical_slot(slot_id: int) -> None:
    """Reject reads outside the physical slot address space."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        ValueError,
        match="slot_ids must be within the configured physical slots",
    ):
        storage.gather(
            layer_index=0,
            slot_ids=(slot_id,),
        )


def test_scatter_rejects_incorrect_value_dtype() -> None:
    """Reject value tensors whose dtype differs from the cache."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((1, 2, 8), dtype=torch.float32)
    values = torch.ones((1, 2, 8), dtype=torch.float16)

    with pytest.raises(
        ValueError,
        match="values dtype must match the KV-cache dtype",
    ):
        storage.scatter(
            layer_index=0,
            slot_ids=(0,),
            keys=keys,
            values=values,
        )


def test_scatter_rejects_incorrect_key_device() -> None:
    """Reject key tensors whose device differs from the cache."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.empty((1, 2, 8), device="meta")
    values = torch.ones((1, 2, 8))

    with pytest.raises(
        ValueError,
        match="keys device must match the KV-cache device",
    ):
        storage.scatter(
            layer_index=0,
            slot_ids=(0,),
            keys=keys,
            values=values,
        )


def test_scatter_rejects_incorrect_value_device() -> None:
    """Reject value tensors whose device differs from the cache."""
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    keys = torch.ones((1, 2, 8))
    values = torch.empty((1, 2, 8), device="meta")

    with pytest.raises(
        ValueError,
        match="values device must match the KV-cache device",
    ):
        storage.scatter(
            layer_index=0,
            slot_ids=(0,),
            keys=keys,
            values=values,
        )
