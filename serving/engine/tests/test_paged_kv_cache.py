"""Tests for phase-aware paged KV-cache operations."""

from typing import cast

import pytest
import torch

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.request import Request
from serving.engine.sequence import SequenceState


def make_storage() -> PagedKVStorage:
    """Create paged KV storage for cache behavior tests."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    storage.key_cache.zero_()
    storage.value_cache.zero_()

    return storage


def make_sequence(
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: tuple[int, ...] = (),
) -> SequenceState:
    """Create a sequence with optional generated history."""
    sequence = SequenceState(
        request=Request(
            request_id="request-1",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=8,
        )
    )
    sequence.generated_token_ids.extend(generated_token_ids)

    return sequence


# Prefill


def test_prefill_writes_and_gathers_complete_history() -> None:
    """Write and gather every logical position during prefill."""
    storage = make_storage()
    cache = PagedKVCache(storage=storage)

    sequence = make_sequence(prompt_token_ids=(1, 2, 3))

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 6, 10),
    )

    keys = torch.arange(
        3 * 2 * 4,
        dtype=torch.float32,
    ).reshape(3, 2, 4)
    values = keys + 100.0

    gathered_keys, gathered_values = cache.update_and_gather(
        layer_index=0,
        execution=execution,
        keys=keys,
        values=values,
    )

    torch.testing.assert_close(gathered_keys, keys)
    torch.testing.assert_close(gathered_values, values)

    torch.testing.assert_close(storage.key_cache[0, 1], keys[0])
    torch.testing.assert_close(storage.key_cache[0, 6], keys[1])
    torch.testing.assert_close(storage.key_cache[0, 10], keys[2])


# Decode


def test_decode_writes_only_newest_slot_and_gathers_complete_history() -> None:
    """Write the newest token during decode while gathering full KV history."""
    storage = make_storage()
    cache = PagedKVCache(storage=storage)

    historical_slots = (1, 6, 10)

    historical_keys = torch.arange(
        3 * 2 * 4,
        dtype=torch.float32,
    ).reshape(3, 2, 4)
    historical_values = historical_keys + 100.0

    storage.scatter(
        layer_index=0,
        slot_ids=historical_slots,
        keys=historical_keys,
        values=historical_values,
    )

    sequence = make_sequence(
        prompt_token_ids=(1, 2, 3),
        generated_token_ids=(4,),
    )

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(1, 6, 10, 14),
    )

    new_keys = torch.full((1, 2, 4), 500.0)
    new_values = torch.full((1, 2, 4), 600.0)

    gathered_keys, gathered_values = cache.update_and_gather(
        layer_index=0,
        execution=execution,
        keys=new_keys,
        values=new_values,
    )

    expected_keys = torch.cat((historical_keys, new_keys), dim=0)
    expected_values = torch.cat((historical_values, new_values), dim=0)

    torch.testing.assert_close(gathered_keys, expected_keys)
    torch.testing.assert_close(gathered_values, expected_values)


# Re-prefill


def test_reprefill_writes_complete_history_with_generated_tokens() -> None:
    """Rewrite complete history when prefill includes generated tokens."""
    storage = make_storage()
    cache = PagedKVCache(storage=storage)

    sequence = make_sequence(
        prompt_token_ids=(1, 2),
        generated_token_ids=(3, 4),
    )

    assert sequence.num_generated_tokens == 2

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(2, 5, 9, 13),
    )

    keys = torch.arange(
        4 * 2 * 4,
        dtype=torch.float32,
    ).reshape(4, 2, 4)
    values = keys + 1000.0

    gathered_keys, gathered_values = cache.update_and_gather(
        layer_index=1,
        execution=execution,
        keys=keys,
        values=values,
    )

    torch.testing.assert_close(gathered_keys, keys)
    torch.testing.assert_close(gathered_values, values)

    for logical_index, slot_id in enumerate(execution.slot_ids):
        torch.testing.assert_close(
            storage.key_cache[1, slot_id],
            keys[logical_index],
        )
        torch.testing.assert_close(
            storage.value_cache[1, slot_id],
            values[logical_index],
        )


# Update ordering


def test_decode_scatters_newest_token_before_gathering_history() -> None:
    """Make the newest decode KV visible in the returned attention history."""
    storage = make_storage()
    cache = PagedKVCache(storage=storage)

    storage.key_cache[0, 1].fill_(1.0)
    storage.key_cache[0, 5].fill_(2.0)
    storage.value_cache[0, 1].fill_(11.0)
    storage.value_cache[0, 5].fill_(12.0)

    sequence = make_sequence(
        prompt_token_ids=(1, 2),
        generated_token_ids=(3,),
    )

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(1, 5, 9),
    )

    new_keys = torch.full((1, 2, 4), 3.0)
    new_values = torch.full((1, 2, 4), 13.0)

    gathered_keys, gathered_values = cache.update_and_gather(
        layer_index=0,
        execution=execution,
        keys=new_keys,
        values=new_values,
    )

    assert torch.all(gathered_keys[-1] == 3.0)
    assert torch.all(gathered_values[-1] == 13.0)


# Validation


def test_update_and_gather_rejects_unsupported_execution_phase() -> None:
    """Reject execution metadata containing an unsupported phase."""
    storage = make_storage()
    cache = PagedKVCache(storage=storage)

    sequence = make_sequence(prompt_token_ids=(1,))

    execution = ModelExecution(
        sequence=sequence,
        phase=cast(ExecutionPhase, "unsupported"),
        slot_ids=(0,),
    )

    keys = torch.ones((1, 2, 4))
    values = torch.ones((1, 2, 4))

    with pytest.raises(
        ValueError,
        match="Unsupported execution phase: unsupported",
    ):
        cache.update_and_gather(
            layer_index=0,
            execution=execution,
            keys=keys,
            values=values,
        )
