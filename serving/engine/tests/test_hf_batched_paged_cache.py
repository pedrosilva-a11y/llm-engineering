"""Tests for the batched Hugging Face paged-cache adapter."""

import pytest
import torch

from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase
from serving.engine.hf_batched_paged_cache import HFBatchedPagedCache
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.tests.helpers import make_execution

NUM_KV_HEADS = 2
HEAD_DIM = 4


def make_paged_cache() -> PagedKVCache:
    """Build a small CPU paged KV cache for adapter tests."""
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=8,
        block_size=4,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    return PagedKVCache(storage)


def hf_states_from_token_values(
    token_values: tuple[float, ...],
) -> torch.Tensor:
    """Build HF-shaped states with one distinct value per token."""
    storage_states = torch.stack(
        [
            torch.full(
                (NUM_KV_HEADS, HEAD_DIM),
                value,
                dtype=torch.float32,
            )
            for value in token_values
        ]
    )

    return storage_states.transpose(0, 1).unsqueeze(0)


def storage_states_from_token_values(
    token_values: tuple[float, ...],
) -> torch.Tensor:
    """Build paged-storage-shaped states with one value per token."""
    return torch.stack(
        [
            torch.full(
                (NUM_KV_HEADS, HEAD_DIM),
                value,
                dtype=torch.float32,
            )
            for value in token_values
        ]
    )


def test_update_single_prefill_preserves_hf_tensor_geometry() -> None:
    """Write and gather one prefill through the batched adapter."""
    paged_cache = make_paged_cache()

    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11, 12),
        phase=ExecutionPhase.PREFILL,
        slot_start=0,
    )

    layout = BatchLayout(executions=(execution,))

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=layout,
    )

    key_states = hf_states_from_token_values((1.0, 2.0, 3.0))
    value_states = hf_states_from_token_values((101.0, 102.0, 103.0))

    keys, values = cache.update(
        key_states=key_states,
        value_states=value_states,
        layer_idx=0,
    )

    assert keys.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
    assert values.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)

    torch.testing.assert_close(keys, key_states)
    torch.testing.assert_close(values, value_states)


def test_update_mixed_decode_and_prefill_builds_packed_history() -> None:
    """Combine cached decode history and a fresh prefill into packed K/V."""
    paged_cache = make_paged_cache()

    decode = make_execution(
        request_id="decode",
        prompt_token_ids=(10, 11, 12),
        generated_token_ids=(13,),
        phase=ExecutionPhase.DECODE,
        slot_start=0,
    )

    prefill = make_execution(
        request_id="prefill",
        prompt_token_ids=(20, 21, 22),
        phase=ExecutionPhase.PREFILL,
        slot_start=8,
    )

    layout = BatchLayout(
        executions=(
            decode,
            prefill,
        )
    )

    assert layout.query_lengths == (1, 3)
    assert layout.key_lengths == (4, 3)
    assert layout.total_query_tokens == 4
    assert layout.total_key_tokens == 7

    storage = paged_cache.storage

    decode_prefix_keys = storage_states_from_token_values((1.0, 2.0, 3.0))
    decode_prefix_values = storage_states_from_token_values((101.0, 102.0, 103.0))

    storage.scatter(
        layer_index=0,
        slot_ids=decode.slot_ids[:-1],
        keys=decode_prefix_keys,
        values=decode_prefix_values,
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=layout,
    )

    # Packed Q region:
    #
    #   decode:  [4]
    #   prefill: [5, 6, 7]
    #
    # Q_total = 4.
    key_states = hf_states_from_token_values((4.0, 5.0, 6.0, 7.0))
    value_states = hf_states_from_token_values((104.0, 105.0, 106.0, 107.0))

    keys, values = cache.update(
        key_states=key_states,
        value_states=value_states,
        layer_idx=0,
    )

    # Packed K region after the update:
    #
    #   decode:  [1, 2, 3, 4]
    #   prefill: [5, 6, 7]
    #
    # K_total = 7.
    expected_keys = hf_states_from_token_values((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

    expected_values = hf_states_from_token_values(
        (
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
        )
    )

    assert keys.shape == (1, NUM_KV_HEADS, 7, HEAD_DIM)
    assert values.shape == (1, NUM_KV_HEADS, 7, HEAD_DIM)

    torch.testing.assert_close(keys, expected_keys)
    torch.testing.assert_close(values, expected_values)


def test_update_writes_each_execution_to_its_physical_slots() -> None:
    """Preserve independent physical slot mappings inside one packed update."""
    paged_cache = make_paged_cache()

    first = make_execution(
        request_id="first",
        prompt_token_ids=(10, 11),
        phase=ExecutionPhase.PREFILL,
        slot_start=1,
    )

    second = make_execution(
        request_id="second",
        prompt_token_ids=(20, 21, 22),
        phase=ExecutionPhase.PREFILL,
        slot_start=10,
    )

    layout = BatchLayout(
        executions=(
            first,
            second,
        )
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=layout,
    )

    key_states = hf_states_from_token_values((1.0, 2.0, 3.0, 4.0, 5.0))

    value_states = hf_states_from_token_values((101.0, 102.0, 103.0, 104.0, 105.0))

    cache.update(
        key_states=key_states,
        value_states=value_states,
        layer_idx=1,
    )

    first_keys, first_values = paged_cache.storage.gather(
        layer_index=1,
        slot_ids=first.slot_ids,
    )

    second_keys, second_values = paged_cache.storage.gather(
        layer_index=1,
        slot_ids=second.slot_ids,
    )

    torch.testing.assert_close(
        first_keys,
        storage_states_from_token_values((1.0, 2.0)),
    )
    torch.testing.assert_close(
        first_values,
        storage_states_from_token_values((101.0, 102.0)),
    )

    torch.testing.assert_close(
        second_keys,
        storage_states_from_token_values((3.0, 4.0, 5.0)),
    )
    torch.testing.assert_close(
        second_values,
        storage_states_from_token_values((103.0, 104.0, 105.0)),
    )


def test_update_supports_multiple_decode_histories() -> None:
    """Gather unequal decode histories on a nonzero transformer layer."""
    paged_cache = make_paged_cache()

    first = make_execution(
        request_id="first",
        prompt_token_ids=(10, 11),
        generated_token_ids=(12,),
        phase=ExecutionPhase.DECODE,
        slot_start=0,
    )

    second = make_execution(
        request_id="second",
        prompt_token_ids=(20, 21, 22, 23),
        generated_token_ids=(24,),
        phase=ExecutionPhase.DECODE,
        slot_start=8,
    )

    layout = BatchLayout(
        executions=(
            first,
            second,
        )
    )

    assert layout.query_lengths == (1, 1)
    assert layout.key_lengths == (3, 5)

    storage = paged_cache.storage
    layer_idx = 1

    storage.scatter(
        layer_index=layer_idx,
        slot_ids=first.slot_ids[:-1],
        keys=storage_states_from_token_values((1.0, 2.0)),
        values=storage_states_from_token_values((101.0, 102.0)),
    )

    storage.scatter(
        layer_index=layer_idx,
        slot_ids=second.slot_ids[:-1],
        keys=storage_states_from_token_values((10.0, 11.0, 12.0, 13.0)),
        values=storage_states_from_token_values((110.0, 111.0, 112.0, 113.0)),
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=layout,
    )

    # One newest query token per decode execution.
    key_states = hf_states_from_token_values((3.0, 14.0))
    value_states = hf_states_from_token_values((103.0, 114.0))

    keys, values = cache.update(
        key_states=key_states,
        value_states=value_states,
        layer_idx=layer_idx,
    )

    expected_keys = hf_states_from_token_values(
        (
            1.0,
            2.0,
            3.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
        )
    )

    expected_values = hf_states_from_token_values(
        (
            101.0,
            102.0,
            103.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
        )
    )

    assert keys.shape == (1, NUM_KV_HEADS, 8, HEAD_DIM)
    assert values.shape == (1, NUM_KV_HEADS, 8, HEAD_DIM)

    torch.testing.assert_close(keys, expected_keys)
    torch.testing.assert_close(values, expected_values)


def test_update_rejects_non_four_dimensional_keys() -> None:
    """Reject K/V tensors outside the Hugging Face cache contract."""
    paged_cache = make_paged_cache()

    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11),
        phase=ExecutionPhase.PREFILL,
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=BatchLayout(executions=(execution,)),
    )

    key_states = torch.zeros(
        (NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=torch.float32,
    )

    value_states = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="key_states must have shape",
    ):
        cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=0,
        )


def test_update_rejects_mismatched_key_value_shapes() -> None:
    """Require key and value tensors to describe identical geometry."""
    paged_cache = make_paged_cache()

    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11),
        phase=ExecutionPhase.PREFILL,
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=BatchLayout(executions=(execution,)),
    )

    key_states = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=torch.float32,
    )

    value_states = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM + 1),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have matching shapes",
    ):
        cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=0,
        )


def test_update_rejects_multiple_hf_batch_rows() -> None:
    """Packed execution lives inside one Hugging Face batch row."""
    paged_cache = make_paged_cache()

    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11),
        phase=ExecutionPhase.PREFILL,
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=BatchLayout(executions=(execution,)),
    )

    key_states = torch.zeros(
        (2, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=torch.float32,
    )

    value_states = torch.zeros_like(key_states)

    with pytest.raises(
        ValueError,
        match="one Hugging Face batch row",
    ):
        cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=0,
        )


def test_update_rejects_wrong_total_query_length() -> None:
    """Require model K/V query tokens to match BatchLayout exactly."""
    paged_cache = make_paged_cache()

    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11, 12),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=layout,
    )

    key_states = torch.zeros(
        (1, NUM_KV_HEADS, 2, HEAD_DIM),
        dtype=torch.float32,
    )

    value_states = torch.zeros_like(key_states)

    with pytest.raises(
        ValueError,
        match="Packed KV query length does not match BatchLayout",
    ):
        cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=0,
        )


def test_scalar_cache_geometry_is_not_exposed() -> None:
    """Reject scalar cache geometry for independent packed sequences."""
    paged_cache = make_paged_cache()

    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11),
        phase=ExecutionPhase.PREFILL,
    )

    cache = HFBatchedPagedCache(
        paged_cache=paged_cache,
        layout=BatchLayout(executions=(execution,)),
    )

    with pytest.raises(RuntimeError, match="scalar sequence length"):
        cache.get_seq_length()

    with pytest.raises(RuntimeError, match="scalar query offset"):
        cache.get_query_offset()

    with pytest.raises(RuntimeError, match="explicit packed attention mask"):
        cache.get_mask_sizes(
            query_length=2,
            layer_idx=0,
        )
