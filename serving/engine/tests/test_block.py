"""Tests for paged KV-cache block allocation primitives."""

import pytest

from serving.cost_models.catalog import TOY_DECODER_MODEL
from serving.engine.block import (
    BlockPool,
    BlockTable,
    blocks_needed,
    kv_block_bytes,
)


def assert_pool_invariant(pool: BlockPool) -> None:
    """Assert consistency between block refcounts and free-queue membership."""
    free_block_ids = set(pool._free_block_ids)

    assert len(free_block_ids) == len(pool._free_block_ids)

    for block in pool._blocks:
        assert (block.ref_count == 0) == (block.block_id in free_block_ids)


# KV Block Sizing


def test_kv_block_bytes_matches_per_token_storage() -> None:
    """Compute block storage as KV bytes per token times block size."""
    block_size = 10
    bytes_per_element = 2

    expected_bytes = block_size * TOY_DECODER_MODEL.kv_bytes_per_token(
        bytes_per_value=bytes_per_element,
    )

    actual_bytes = kv_block_bytes(
        model=TOY_DECODER_MODEL,
        block_size=block_size,
        bytes_per_element=bytes_per_element,
    )

    assert actual_bytes == expected_bytes


@pytest.mark.parametrize(
    ("block_size", "bytes_per_element"),
    [
        (0, 2),
        (-1, 2),
        (16, 0),
        (16, -1),
    ],
)
def test_kv_block_bytes_rejects_non_positive_values(
    block_size: int,
    bytes_per_element: int,
) -> None:
    """Reject non-positive block sizes and element sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        kv_block_bytes(
            model=TOY_DECODER_MODEL,
            block_size=block_size,
            bytes_per_element=bytes_per_element,
        )


# Block-count helpers


@pytest.mark.parametrize(
    ("num_tokens", "block_size", "expected_blocks"),
    [
        (0, 16, 0),
        (1, 16, 1),
        (15, 16, 1),
        (16, 16, 1),
        (17, 16, 2),
        (32, 16, 2),
        (33, 16, 3),
    ],
)
def test_blocks_needed_rounds_up_to_full_blocks(
    num_tokens: int,
    block_size: int,
    expected_blocks: int,
) -> None:
    """Round token storage requirements up to whole physical blocks."""
    actual_blocks = blocks_needed(
        num_tokens=num_tokens,
        block_size=block_size,
    )

    assert actual_blocks == expected_blocks


@pytest.mark.parametrize(
    ("num_tokens", "block_size"),
    [
        (-1, 16),
        (16, 0),
        (16, -1),
    ],
)
def test_blocks_needed_rejects_invalid_values(
    num_tokens: int,
    block_size: int,
) -> None:
    """Reject negative token counts and non-positive block sizes."""
    with pytest.raises(ValueError):
        blocks_needed(
            num_tokens=num_tokens,
            block_size=block_size,
        )


# BlockPool Initialization


@pytest.mark.parametrize(
    "num_blocks",
    [
        0,
        -1,
    ],
)
def test_block_pool_rejects_non_positive_block_count(
    num_blocks: int,
) -> None:
    """Reject pools with a non-positive physical block count."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        BlockPool(num_blocks=num_blocks)


def test_block_pool_initializes_with_all_blocks_free() -> None:
    """Initialize every physical block as free and available for allocation."""
    pool = BlockPool(num_blocks=4)

    assert pool.num_free_blocks == 4
    assert pool.num_allocated_blocks == 0
    assert pool.utilization == 0.0

    assert all(block.ref_count == 0 for block in pool._blocks)

    assert tuple(pool._free_block_ids) == (0, 1, 2, 3)

    assert_pool_invariant(pool)


# BlockPool allocation


def test_allocate_returns_distinct_blocks_in_fifo_order() -> None:
    """Allocate distinct physical block identifiers in free-queue order."""
    pool = BlockPool(num_blocks=4)
    allocated_block_ids = pool.allocate(num_blocks=3)

    assert allocated_block_ids == (0, 1, 2)
    assert len(set(allocated_block_ids)) == 3
    assert tuple(pool._free_block_ids) == (3,)

    assert_pool_invariant(pool)


def test_allocate_updates_pool_capacity() -> None:
    """Update free, allocated, and utilization values after allocation."""
    pool = BlockPool(num_blocks=4)
    allocated_block_ids = pool.allocate(num_blocks=2)

    assert pool.num_free_blocks == 2
    assert pool.num_allocated_blocks == 2
    assert pool.utilization == 0.5

    assert all(
        pool._blocks[block_id].ref_count == 1 for block_id in allocated_block_ids
    )

    assert_pool_invariant(pool)


@pytest.mark.parametrize(
    "num_blocks",
    [
        0,
        -1,
    ],
)
def test_allocate_rejects_non_positive_block_count(
    num_blocks: int,
) -> None:
    """Reject allocation requests for zero or a negative number of blocks."""
    pool = BlockPool(num_blocks=4)

    with pytest.raises(
        ValueError,
        match="num_blocks must be greater than zero",
    ):
        pool.allocate(num_blocks=num_blocks)


def test_allocate_fails_atomically_when_capacity_is_insufficient() -> None:
    """Leave pool state unchanged when an allocation cannot be satisfied."""
    pool = BlockPool(num_blocks=4)
    first_allocation = pool.allocate(num_blocks=2)

    free_before = tuple(pool._free_block_ids)
    ref_counts_before = tuple(block.ref_count for block in pool._blocks)

    with pytest.raises(RuntimeError, match="Not enough free blocks available"):
        pool.allocate(num_blocks=3)

    assert first_allocation == (0, 1)
    assert tuple(pool._free_block_ids) == free_before
    assert tuple(block.ref_count for block in pool._blocks) == ref_counts_before
    assert pool.num_free_blocks == 2
    assert pool.num_allocated_blocks == 2

    assert_pool_invariant(pool)


# BlockPool reference counting


def test_increment_ref_count_adds_reference_to_allocated_blocks() -> None:
    """Increase the reference count of an already allocated block."""
    pool = BlockPool(num_blocks=4)

    block_id = pool.allocate(num_blocks=1)[0]

    pool.increment_ref_count(block_id=block_id)

    assert pool._blocks[block_id].ref_count == 2
    assert pool.num_allocated_blocks == 1
    assert block_id not in pool._free_block_ids

    assert_pool_invariant(pool)


@pytest.mark.parametrize(
    "block_id",
    [
        -1,
        4,
    ],
)
def test_increment_ref_count_rejects_invalid_block(
    block_id: int,
) -> None:
    """Reject reference increments for blocks outside the pool."""
    pool = BlockPool(num_blocks=4)

    with pytest.raises(
        ValueError,
        match="Invalid block_id",
    ):
        pool.increment_ref_count(block_id)


def test_increment_ref_count_rejects_free_block() -> None:
    """Reject adding a reference to a currently free physical block."""
    pool = BlockPool(num_blocks=4)

    with pytest.raises(
        ValueError,
        match="free block",
    ):
        pool.increment_ref_count(block_id=0)

    assert_pool_invariant(pool)


# BlockPool release


def test_free_keeps_shared_block_allocated_until_final_reference() -> None:
    """Keep a shared block allocated until its last reference is released."""
    pool = BlockPool(num_blocks=4)

    block_id = pool.allocate(num_blocks=1)[0]
    pool.increment_ref_count(block_id=block_id)

    assert pool._blocks[block_id].ref_count == 2

    pool.free(block_id=block_id)

    assert pool._blocks[block_id].ref_count == 1
    assert block_id not in pool._free_block_ids
    assert pool.num_allocated_blocks == 1

    pool.free(block_id=block_id)

    assert pool._blocks[block_id].ref_count == 0
    assert block_id in pool._free_block_ids
    assert pool.num_allocated_blocks == 0

    assert_pool_invariant(pool)


def test_free_returns_block_to_end_of_fifo_queue() -> None:
    """Return a fully released block to the end of the free-block queue."""
    pool = BlockPool(num_blocks=4)

    allocated_block_ids = pool.allocate(num_blocks=2)

    first_block_id = allocated_block_ids[0]

    pool.free(first_block_id)

    assert tuple(pool._free_block_ids) == (
        2,
        3,
        0,
    )

    next_allocation = pool.allocate(num_blocks=1)

    assert next_allocation == (2,)

    assert_pool_invariant(pool)


@pytest.mark.parametrize(
    "block_id",
    [
        -1,
        4,
    ],
)
def test_free_rejects_invalid_block(
    block_id: int,
) -> None:
    """Reject releasing physical blocks outside the pool."""
    pool = BlockPool(num_blocks=4)

    with pytest.raises(
        ValueError,
        match="Invalid block_id",
    ):
        pool.free(block_id)


def test_free_rejects_already_free_block() -> None:
    """Reject releasing a physical block that is already free."""
    pool = BlockPool(num_blocks=4)

    with pytest.raises(
        ValueError,
        match="already free",
    ):
        pool.free(block_id=0)

    assert_pool_invariant(pool)


def test_free_many_releases_multiple_blocks() -> None:
    """Release one reference from each supplied physical block."""
    pool = BlockPool(num_blocks=4)

    allocated_block_ids = pool.allocate(num_blocks=3)

    pool.free_many(allocated_block_ids)

    assert pool.num_free_blocks == 4
    assert pool.num_allocated_blocks == 0
    assert pool.utilization == 0.0

    assert all(block.ref_count == 0 for block in pool._blocks)

    assert tuple(pool._free_block_ids) == (
        3,
        0,
        1,
        2,
    )

    assert_pool_invariant(pool)


# BlockPool invariants


def test_pool_preserves_free_queue_reference_count_invariant() -> None:
    """Keep free-queue membership consistent with block reference counts."""
    pool = BlockPool(num_blocks=8)

    first_allocation = pool.allocate(num_blocks=4)

    pool.increment_ref_count(first_allocation[0])
    pool.increment_ref_count(first_allocation[1])

    pool.free(first_allocation[2])
    pool.free(first_allocation[0])

    second_allocation = pool.allocate(num_blocks=2)

    pool.free(first_allocation[1])
    pool.free(first_allocation[1])
    pool.free(first_allocation[3])

    pool.increment_ref_count(second_allocation[0])

    assert_pool_invariant(pool)

    assert pool.num_free_blocks + pool.num_allocated_blocks == pool.num_blocks


# BlockTable initialization


@pytest.mark.parametrize(
    "block_size",
    [
        0,
        -1,
    ],
)
def test_block_table_rejects_non_positive_block_size(
    block_size: int,
) -> None:
    """Reject block tables configured with a non-positive block size."""
    pool = BlockPool(num_blocks=4)

    with pytest.raises(ValueError, match="must be greater than zero"):
        BlockTable(pool=pool, block_size=block_size)


# BlockTable request allocation


def test_allocate_for_request_records_required_blocks() -> None:
    """Allocate and record the blocks required by a request."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    block_ids = table.allocate_for_request(
        request_id="request-1",
        num_tokens=17,
    )

    assert block_ids == (0, 1)
    assert table.blocks_for_request("request-1") == (0, 1)

    assert pool.num_allocated_blocks == 2
    assert pool.num_free_blocks == 6

    assert_pool_invariant(pool)


def test_allocate_for_request_rejects_duplicate_request_id() -> None:
    """Reject allocating a second block table for the same request."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    first_block_ids = table.allocate_for_request(
        request_id="request-1",
        num_tokens=16,
    )

    with pytest.raises(ValueError, match="already has a block table"):
        table.allocate_for_request(
            request_id="request-1",
            num_tokens=16,
        )

    assert table.blocks_for_request("request-1") == first_block_ids
    assert pool.num_allocated_blocks == 1
    assert pool.num_free_blocks == 7

    assert_pool_invariant(pool)


@pytest.mark.parametrize(
    "num_tokens",
    [
        0,
        -1,
    ],
)
def test_allocate_for_request_rejects_non_positive_token_count(
    num_tokens: int,
) -> None:
    """Reject request allocation for a non-positive token count."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    with pytest.raises(ValueError, match="must be greater than zero"):
        table.allocate_for_request(
            request_id="request-1",
            num_tokens=num_tokens,
        )

    assert pool.num_allocated_blocks == 0

    assert_pool_invariant(pool)


def test_allocate_for_request_propagates_pool_exhaustion_atomically() -> None:
    """Leave request-table state unchanged when pool capacity is insufficient."""
    pool = BlockPool(num_blocks=2)
    table = BlockTable(pool=pool, block_size=16)

    with pytest.raises(RuntimeError, match="Not enough free blocks available"):
        table.allocate_for_request(request_id="request-1", num_tokens=33)

    assert pool.num_free_blocks == 2
    assert pool.num_allocated_blocks == 0
    assert "request-1" not in table._block_tables

    assert_pool_invariant(pool)


# BlockTable lookup and release


def test_blocks_for_request_returns_blocks_in_logical_order() -> None:
    """Return an immutable view of a request's physical block mapping."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=33)

    block_ids = table.blocks_for_request(request_id=request_id)

    assert block_ids == (0, 1, 2)
    assert isinstance(block_ids, tuple)


def test_blocks_for_request_rejects_unknown_request() -> None:
    """Reject block lookup for an unknown request identifier."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    with pytest.raises(ValueError, match="Unknown request_id"):
        table.blocks_for_request("unknown-request")


def test_free_request_releases_all_blocks_and_removes_mapping() -> None:
    """Release a request's physical blocks and remove its block-table entry."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    block_ids = table.allocate_for_request(request_id=request_id, num_tokens=33)

    assert block_ids == (0, 1, 2)
    assert pool.num_allocated_blocks == 3
    assert pool.num_free_blocks == 5

    table.free_request(request_id)

    assert pool.num_allocated_blocks == 0
    assert pool.num_free_blocks == 8
    assert request_id not in table._block_tables

    assert_pool_invariant(pool)


def test_free_request_preserves_other_request_blocks() -> None:
    """Release one request without affecting another request's block mapping."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    first_request_id = "request-1"
    second_request_id = "request-2"

    first_blocks = table.allocate_for_request(
        request_id=first_request_id,
        num_tokens=17,
    )
    second_blocks = table.allocate_for_request(
        request_id=second_request_id,
        num_tokens=16,
    )

    assert first_blocks == (0, 1)
    assert second_blocks == (2,)
    assert pool.num_allocated_blocks == 3

    table.free_request(first_request_id)

    assert second_request_id in table._block_tables
    assert table.blocks_for_request(second_request_id) == second_blocks

    assert pool._blocks[second_blocks[0]].ref_count == 1
    assert second_blocks[0] not in pool._free_block_ids

    assert pool.num_allocated_blocks == 1
    assert pool.num_free_blocks == 7

    assert_pool_invariant(pool)


def test_free_request_rejects_unknown_request() -> None:
    """Reject releasing blocks for an unknown request identifier."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    with pytest.raises(ValueError, match="Unknown request_id"):
        table.free_request("unknown-request")

    assert_pool_invariant(pool)


# BlockTable incremental growth


def test_append_token_within_existing_block_allocates_nothing() -> None:
    """Avoid allocating a block when the token position is already mapped."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=1)

    free_blocks_before = pool.num_free_blocks

    table.append_token(request_id=request_id, token_position=15)

    assert table.blocks_for_request(request_id) == (0,)
    assert pool.num_free_blocks == free_blocks_before

    assert_pool_invariant(pool)


def test_append_token_allocates_at_block_boundary() -> None:
    """Allocate exactly one new block when crossing a block boundary."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    table.append_token(request_id=request_id, token_position=16)

    assert table.blocks_for_request(request_id) == (0, 1)
    assert pool.num_allocated_blocks == 2

    assert_pool_invariant(pool)


def test_append_token_fails_atomically_when_pool_is_exhausted() -> None:
    """Leave request and pool state unchanged when append allocation fails."""
    pool = BlockPool(num_blocks=2)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    pool.allocate(num_blocks=1)

    blocks_before = table.blocks_for_request(request_id=request_id)
    free_blocks_before = tuple(pool._free_block_ids)
    ref_counts_before = tuple(block.ref_count for block in pool._blocks)

    with pytest.raises(RuntimeError, match="Not enough free blocks available"):
        table.append_token(
            request_id=request_id,
            token_position=16,
        )

    assert table.blocks_for_request(request_id) == blocks_before
    assert tuple(pool._free_block_ids) == free_blocks_before
    assert tuple(block.ref_count for block in pool._blocks) == ref_counts_before

    assert pool.num_free_blocks == 0
    assert pool.num_allocated_blocks == 2

    assert_pool_invariant(pool)


def test_append_token_grows_across_two_boundaries() -> None:
    """Grow the block table correctly across multiple block boundaries."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    table.append_token(request_id=request_id, token_position=16)

    table.append_token(request_id=request_id, token_position=32)

    assert table.blocks_for_request(request_id) == (0, 1, 2)
    assert pool.num_allocated_blocks == 3

    assert_pool_invariant(pool)


def test_append_token_is_idempotent_for_mapped_position() -> None:
    """Avoid duplicate allocation when the same mapped position is repeated."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    table.append_token(request_id=request_id, token_position=16)

    blocks_after_first_append = table.blocks_for_request(request_id)
    free_blocks_after_first_append = pool.num_free_blocks

    table.append_token(request_id=request_id, token_position=16)

    assert table.blocks_for_request(request_id) == blocks_after_first_append
    assert pool.num_free_blocks == free_blocks_after_first_append

    assert_pool_invariant(pool)


def test_append_token_rejects_position_that_skips_logical_blocks() -> None:
    """Reject token positions that jump over unallocated logical blocks."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    with pytest.raises(ValueError, match="skips one or more logical blocks"):
        table.append_token(request_id=request_id, token_position=32)

    assert table.blocks_for_request(request_id) == (0,)
    assert pool.num_allocated_blocks == 1

    assert_pool_invariant(pool)


def test_append_token_rejects_negative_position() -> None:
    """Reject negative logical token positions."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=1)

    with pytest.raises(ValueError, match="token_position must not be negative"):
        table.append_token(request_id=request_id, token_position=-1)

    assert_pool_invariant(pool)


def test_append_token_rejects_unknown_request() -> None:
    """Reject appending storage for an unknown request identifier."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    with pytest.raises(ValueError, match="Unknown request_id"):
        table.append_token(request_id="unknown-request", token_position=0)

    assert_pool_invariant(pool)


# Slot mapping


def test_slot_for_position_maps_logical_position_to_physical_slot() -> None:
    """Map logical token positions to their flat physical KV-cache slots."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=17)

    assert table.slot_for_position(request_id, 0) == 0
    assert table.slot_for_position(request_id, 15) == 15
    assert table.slot_for_position(request_id, 16) == 16


def test_slot_for_position_supports_non_contiguous_physical_blocks() -> None:
    """Resolve slots correctly when logical blocks map to non-contiguous blocks."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    occupied_block_ids = pool.allocate(num_blocks=2)

    assert occupied_block_ids == (1, 2)

    table.append_token(request_id=request_id, token_position=16)

    assert table.blocks_for_request(request_id) == (0, 3)

    assert table.slot_for_position(request_id, 15) == 15
    assert table.slot_for_position(request_id, 16) == 48
    assert table.slot_for_position(request_id, 17) == 49

    assert_pool_invariant(pool)


def test_slot_for_position_rejects_unmapped_position() -> None:
    """Reject token positions whose logical block has not been allocated."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    with pytest.raises(ValueError, match="No physical block exists for token position"):
        table.slot_for_position(request_id=request_id, token_position=16)


def test_slot_for_position_rejects_negative_position() -> None:
    """Reject negative logical positions during slot lookup."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=1)

    with pytest.raises(ValueError, match="token_position must not be negative"):
        table.slot_for_position(request_id=request_id, token_position=-1)


def test_slot_for_position_rejects_unknown_request() -> None:
    """Reject slot lookup for an unknown request identifier."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    with pytest.raises(ValueError, match="Unknown request_id"):
        table.slot_for_position(request_id="unknown-request", token_position=0)


# Fragmentation


@pytest.mark.parametrize(
    ("num_tokens", "expected_fragmentation"),
    [
        (1, 15),
        (17, 15),
        (31, 1),
    ],
)
def test_fragmentation_slots_reports_internal_waste(
    num_tokens: int,
    expected_fragmentation: int,
) -> None:
    """Report unused token slots in partially filled physical blocks."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=num_tokens)

    fragmentation = table.fragmentation_slots(
        request_id=request_id,
        num_tokens=num_tokens,
    )

    assert fragmentation == expected_fragmentation


@pytest.mark.parametrize(
    "num_tokens",
    [
        16,
        32,
        48,
        64,
    ],
)
def test_fragmentation_slots_reports_zero_for_full_blocks(
    num_tokens: int,
) -> None:
    """Report zero fragmentation when allocated blocks are completely filled."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=num_tokens)

    fragmentation = table.fragmentation_slots(
        request_id=request_id,
        num_tokens=num_tokens,
    )

    assert fragmentation == 0


def test_fragmentation_slots_rejects_inconsistent_token_count() -> None:
    """Reject token counts that do not match the request's allocated capacity."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(pool=pool, block_size=16)

    request_id = "request-1"

    table.allocate_for_request(request_id=request_id, num_tokens=16)

    with pytest.raises(ValueError, match="inconsistent with the allocated blocks"):
        table.fragmentation_slots(
            request_id=request_id,
            num_tokens=17,
        )


@pytest.mark.parametrize(
    "num_tokens",
    [
        0,
        -1,
    ],
)
def test_fragmentation_slots_rejects_non_positive_token_count(
    num_tokens: int,
) -> None:
    """Reject fragmentation calculations for non-positive token counts."""
    pool = BlockPool(num_blocks=8)
    table = BlockTable(
        pool=pool,
        block_size=16,
    )

    table.allocate_for_request(
        request_id="request-1",
        num_tokens=1,
    )

    with pytest.raises(ValueError, match="num_tokens must be greater than zero"):
        table.fragmentation_slots(
            request_id="request-1",
            num_tokens=num_tokens,
        )


def test_fragmentation_slots_rejects_unknown_request() -> None:
    """Reject fragmentation lookup for an unknown request identifier."""
    pool = BlockPool(num_blocks=4)
    table = BlockTable(
        pool=pool,
        block_size=16,
    )

    with pytest.raises(ValueError, match="Unknown request_id"):
        table.fragmentation_slots(
            request_id="unknown-request",
            num_tokens=1,
        )
