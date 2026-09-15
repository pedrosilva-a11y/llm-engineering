"""Tests for the Hugging Face paged KV-cache adapter."""

from typing import cast

import pytest
import torch

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.hf_paged_cache import HFPagedCache
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.request import Request
from serving.engine.sequence import SequenceState

NUM_LAYERS = 2
NUM_BLOCKS = 2
BLOCK_SIZE = 8
NUM_KV_HEADS = 2
HEAD_DIM = 4
DTYPE = torch.float32
DEVICE = torch.device("cpu")


def make_storage() -> PagedKVStorage:
    """Create a small CPU paged KV storage."""
    return PagedKVStorage(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )


def make_sequence(
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: tuple[int, ...] = (),
) -> SequenceState:
    """Create a sequence with optional generated token history."""
    sequence = SequenceState(
        request=Request(
            request_id="request-1",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=16,
        )
    )

    for token_id in generated_token_ids:
        sequence.append_token(token_id)

    return sequence


def make_execution(
    sequence: SequenceState,
    phase: ExecutionPhase,
    slot_ids: tuple[int, ...],
) -> ModelExecution:
    """Create model execution metadata."""
    return ModelExecution(
        sequence=sequence,
        phase=phase,
        slot_ids=slot_ids,
    )


def make_hf_states(
    num_tokens: int,
    start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic Hugging Face-style key and value tensors."""
    num_values = num_tokens * NUM_KV_HEADS * HEAD_DIM

    keys = torch.arange(
        start,
        start + num_values,
        dtype=DTYPE,
        device=DEVICE,
    ).reshape(
        1,
        NUM_KV_HEADS,
        num_tokens,
        HEAD_DIM,
    )
    values = keys + 1000.0

    return keys, values


# Cache geometry


def test_prefill_cache_geometry() -> None:
    """Return empty past-cache geometry during prefill."""
    sequence = make_sequence(prompt_token_ids=(1, 2, 3))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(5, 1, 9),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    assert cache.get_seq_length() == 0
    assert cache.get_query_offset() == 0
    assert cache.get_mask_sizes(query_length=3, layer_idx=0) == (3, 0)


def test_decode_cache_geometry() -> None:
    """Return existing history geometry during decode."""
    sequence = make_sequence(
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=(4,),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(5, 1, 9, 12),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    assert sequence.current_length == 4
    assert cache.get_seq_length() == 3
    assert cache.get_query_offset() == 3
    assert cache.get_mask_sizes(query_length=1, layer_idx=0) == (4, 0)


def test_reprefill_cache_geometry_ignores_generated_history() -> None:
    """Treat generated history as current input during re-prefill."""
    sequence = make_sequence(
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=(4, 5),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(5, 1, 9, 12, 3),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    assert sequence.current_length == 5
    assert sequence.num_generated_tokens == 2
    assert cache.get_seq_length() == 0
    assert cache.get_query_offset() == 0
    assert cache.get_mask_sizes(query_length=5, layer_idx=0) == (5, 0)


def test_cache_geometry_is_independent_of_layer_index() -> None:
    """Return identical cache geometry for every transformer layer."""
    sequence = make_sequence(
        prompt_token_ids=(1, 2),
        generated_token_ids=(3,),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(2, 8, 13),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    assert cache.get_seq_length(layer_idx=1) == 2
    assert cache.get_query_offset(layer_idx=1) == 2
    assert cache.get_mask_sizes(query_length=1, layer_idx=1) == (3, 0)


# Prefill


def test_prefill_update_writes_and_returns_full_history() -> None:
    """Write and return the complete KV history during prefill."""
    storage = make_storage()
    sequence = make_sequence(prompt_token_ids=(1, 2, 3))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(storage),
        execution=execution,
    )
    keys, values = make_hf_states(num_tokens=3)

    returned_keys, returned_values = cache.update(
        key_states=keys,
        value_states=values,
        layer_idx=1,
    )

    assert torch.equal(returned_keys, keys)
    assert torch.equal(returned_values, values)

    stored_keys, stored_values = storage.gather(
        layer_index=1,
        slot_ids=execution.slot_ids,
    )

    assert torch.equal(
        stored_keys,
        keys[0].transpose(0, 1),
    )
    assert torch.equal(
        stored_values,
        values[0].transpose(0, 1),
    )


# Decode


def test_decode_update_writes_newest_token_and_returns_full_history() -> None:
    """Write one decode token and return the complete KV history."""
    storage = make_storage()
    sequence = make_sequence(
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=(4,),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(7, 2, 11, 5),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(storage),
        execution=execution,
    )

    previous_keys = torch.arange(
        3 * NUM_KV_HEADS * HEAD_DIM,
        dtype=DTYPE,
    ).reshape(
        3,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    previous_values = previous_keys + 1000.0

    storage.scatter(
        layer_index=0,
        slot_ids=execution.slot_ids[:-1],
        keys=previous_keys,
        values=previous_values,
    )

    new_keys, new_values = make_hf_states(
        num_tokens=1,
        start=100,
    )

    returned_keys, returned_values = cache.update(
        key_states=new_keys,
        value_states=new_values,
        layer_idx=0,
    )

    new_storage_keys = new_keys[0].transpose(0, 1)
    new_storage_values = new_values[0].transpose(0, 1)

    expected_storage_keys = torch.cat(
        (previous_keys, new_storage_keys),
        dim=0,
    )
    expected_storage_values = torch.cat(
        (previous_values, new_storage_values),
        dim=0,
    )

    expected_hf_keys = expected_storage_keys.transpose(0, 1).unsqueeze(0)
    expected_hf_values = expected_storage_values.transpose(0, 1).unsqueeze(0)

    assert torch.equal(returned_keys, expected_hf_keys)
    assert torch.equal(returned_values, expected_hf_values)


# Re-prefill


def test_reprefill_update_rewrites_complete_logical_history() -> None:
    """Rewrite and return the complete logical history during re-prefill."""
    storage = make_storage()
    sequence = make_sequence(
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=(4, 5),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11, 5, 14),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(storage),
        execution=execution,
    )

    keys, values = make_hf_states(num_tokens=5)

    returned_keys, returned_values = cache.update(
        key_states=keys,
        value_states=values,
        layer_idx=0,
    )

    assert torch.equal(returned_keys, keys)
    assert torch.equal(returned_values, values)

    stored_keys, stored_values = storage.gather(
        layer_index=0,
        slot_ids=execution.slot_ids,
    )

    assert torch.equal(
        stored_keys,
        keys[0].transpose(0, 1),
    )
    assert torch.equal(
        stored_values,
        values[0].transpose(0, 1),
    )


# Validation


def test_get_mask_sizes_rejects_wrong_prefill_query_length() -> None:
    """Reject a mask query length inconsistent with prefill execution."""
    sequence = make_sequence(prompt_token_ids=(1, 2, 3))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2, 3),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    with pytest.raises(
        ValueError,
        match="query_length does not match",
    ):
        cache.get_mask_sizes(query_length=1, layer_idx=0)


def test_update_rejects_non_four_dimensional_keys() -> None:
    """Reject key states that do not use the Hugging Face cache shape."""
    sequence = make_sequence(prompt_token_ids=(1, 2))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    keys = torch.zeros(
        (NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=DTYPE,
    )
    values = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=DTYPE,
    )

    with pytest.raises(
        ValueError,
        match="key_states must have shape",
    ):
        cache.update(keys, values, layer_idx=0)


def test_update_rejects_non_four_dimensional_values() -> None:
    """Reject value states that do not use the Hugging Face cache shape."""
    sequence = make_sequence(prompt_token_ids=(1, 2))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    keys = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=DTYPE,
    )
    values = torch.zeros(
        (NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=DTYPE,
    )

    with pytest.raises(
        ValueError,
        match="value_states must have shape",
    ):
        cache.update(keys, values, layer_idx=0)


def test_update_rejects_mismatched_key_value_shapes() -> None:
    """Reject key and value states with different shapes."""
    sequence = make_sequence(prompt_token_ids=(1, 2))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    keys = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=DTYPE,
    )
    values = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM + 1),
        dtype=DTYPE,
    )

    with pytest.raises(
        ValueError,
        match="must have matching shapes",
    ):
        cache.update(keys, values, layer_idx=0)


def test_update_rejects_batch_size_greater_than_one() -> None:
    """Reject Hugging Face cache updates with multiple sequences."""
    sequence = make_sequence(prompt_token_ids=(1, 2))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    keys = torch.zeros(
        (2, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=DTYPE,
    )
    values = torch.zeros_like(keys)

    with pytest.raises(
        ValueError,
        match="batch size 1",
    ):
        cache.update(keys, values, layer_idx=0)


def test_prefill_update_rejects_wrong_query_length() -> None:
    """Reject prefill KV states that omit part of the current history."""
    sequence = make_sequence(prompt_token_ids=(1, 2, 3))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2, 3),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    keys, values = make_hf_states(num_tokens=1)

    with pytest.raises(
        ValueError,
        match="KV query length does not match",
    ):
        cache.update(keys, values, layer_idx=0)


def test_decode_update_rejects_more_than_one_query_token() -> None:
    """Reject decode KV states containing more than one query token."""
    sequence = make_sequence(
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=(4,),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(1, 2, 3, 4),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    keys, values = make_hf_states(num_tokens=2)

    with pytest.raises(
        ValueError,
        match="KV query length does not match",
    ):
        cache.update(keys, values, layer_idx=0)


def test_get_seq_length_rejects_unsupported_phase() -> None:
    """Reject cache sequence-length queries for unsupported phases."""
    sequence = make_sequence(prompt_token_ids=(1,))
    execution = make_execution(
        sequence=sequence,
        phase=cast(ExecutionPhase, "unsupported"),
        slot_ids=(1,),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported execution phase",
    ):
        cache.get_seq_length()


def test_get_mask_sizes_rejects_unsupported_phase() -> None:
    """Reject mask-size queries for unsupported execution phases."""
    sequence = make_sequence(prompt_token_ids=(1,))
    execution = make_execution(
        sequence=sequence,
        phase=cast(ExecutionPhase, "unsupported"),
        slot_ids=(1,),
    )
    cache = HFPagedCache(
        paged_cache=PagedKVCache(make_storage()),
        execution=execution,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported execution phase",
    ):
        cache.get_mask_sizes(query_length=1, layer_idx=0)
