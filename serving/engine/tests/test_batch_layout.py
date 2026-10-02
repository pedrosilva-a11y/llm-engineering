"""Tests for packed model-execution batch layout."""

from typing import cast

import pytest

from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase
from serving.engine.tests.helpers import make_execution


def test_rejects_empty_batch() -> None:
    """Reject a layout without model executions."""
    with pytest.raises(ValueError, match="BatchLayout requires at least one execution"):
        BatchLayout(executions=())


def test_rejects_unsupportedmake_execution_phase() -> None:
    """Reject an execution whose phase is not supported."""
    execution = make_execution(
        request_id="unsupported",
        prompt_token_ids=(10, 11, 12),
        phase=cast(ExecutionPhase, "unsupported"),
    )

    with pytest.raises(ValueError, match="Unsupported execution phase: unsupported"):
        BatchLayout(executions=(execution,))


def test_single_prefill_uses_full_sequence_for_queries_and_keys() -> None:
    """Treat a fresh prefill sequence as full query and key history."""
    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11, 12),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))

    assert layout.query_lengths == (3,)
    assert layout.key_lengths == (3,)

    assert layout.query_offsets == (0,)
    assert layout.key_offsets == (0,)

    assert layout.logit_indices == (2,)

    assert layout.total_query_tokens == 3
    assert layout.total_key_tokens == 3


def test_single_decode_uses_one_query_and_full_key_history() -> None:
    """Treat decode as one query attending over the full sequence history."""
    execution = make_execution(
        request_id="decode",
        prompt_token_ids=(20, 21, 22),
        generated_token_ids=(23,),
        phase=ExecutionPhase.DECODE,
    )

    layout = BatchLayout(executions=(execution,))

    assert execution.sequence.current_length == 4

    assert layout.query_lengths == (1,)
    assert layout.key_lengths == (4,)

    assert layout.query_offsets == (0,)
    assert layout.key_offsets == (0,)

    assert layout.logit_indices == (0,)

    assert layout.total_query_tokens == 1
    assert layout.total_key_tokens == 4


def test_all_decode_batch_derives_independent_query_and_key_offsets() -> None:
    """Keep packed query and key spaces independent for decode-only batches."""
    executions = (
        make_execution(
            request_id="a",
            prompt_token_ids=(10, 11, 12, 13),
            generated_token_ids=(14,),
            phase=ExecutionPhase.DECODE,
            slot_start=0,
        ),
        make_execution(
            request_id="b",
            prompt_token_ids=(20, 21, 22, 23, 24, 25),
            generated_token_ids=(26, 27),
            phase=ExecutionPhase.DECODE,
            slot_start=10,
        ),
        make_execution(
            request_id="c",
            prompt_token_ids=(30, 31),
            generated_token_ids=(32,),
            phase=ExecutionPhase.DECODE,
            slot_start=20,
        ),
    )

    layout = BatchLayout(executions=executions)

    assert layout.query_lengths == (1, 1, 1)
    assert layout.key_lengths == (5, 8, 3)

    assert layout.query_offsets == (0, 1, 2)
    assert layout.key_offsets == (0, 5, 13)

    assert layout.logit_indices == (0, 1, 2)

    assert layout.total_query_tokens == 3
    assert layout.total_key_tokens == 16


def test_mixed_batch_derives_expected_packed_geometry() -> None:
    """Derive query and key regions for mixed prefill and decode work."""
    prefill_a = make_execution(
        request_id="a",
        prompt_token_ids=(10, 11, 12),
        phase=ExecutionPhase.PREFILL,
        slot_start=0,
    )

    decode_b = make_execution(
        request_id="b",
        prompt_token_ids=(20, 21, 22),
        generated_token_ids=(23,),
        phase=ExecutionPhase.DECODE,
        slot_start=10,
    )

    prefill_c = make_execution(
        request_id="c",
        prompt_token_ids=(30, 31),
        phase=ExecutionPhase.PREFILL,
        slot_start=20,
    )

    layout = BatchLayout(executions=(prefill_a, decode_b, prefill_c))

    assert layout.query_lengths == (3, 1, 2)
    assert layout.key_lengths == (3, 4, 2)

    assert layout.query_offsets == (0, 3, 4)
    assert layout.key_offsets == (0, 3, 7)

    assert layout.logit_indices == (2, 3, 5)

    assert layout.total_query_tokens == 6
    assert layout.total_key_tokens == 9


def test_reprefill_uses_complete_current_sequence_history() -> None:
    """Use prompt plus generated history when a sequence re-prefilled."""
    execution = make_execution(
        request_id="reprefill",
        prompt_token_ids=(10, 11, 12),
        generated_token_ids=(40, 41),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))

    assert execution.sequence.current_length == 5

    assert layout.query_lengths == (5,)
    assert layout.key_lengths == (5,)

    assert layout.logit_indices == (4,)

    assert layout.total_query_tokens == 5
    assert layout.total_key_tokens == 5


def test_offsets_and_totals_are_self_consistent() -> None:
    """Keep packed offsets consistent with query and key token totals."""
    executions = (
        make_execution(
            request_id="a",
            prompt_token_ids=(1, 2, 3, 4),
            phase=ExecutionPhase.PREFILL,
        ),
        make_execution(
            request_id="b",
            prompt_token_ids=(5, 6),
            generated_token_ids=(7, 8, 9),
            phase=ExecutionPhase.DECODE,
            slot_start=10,
        ),
        make_execution(
            request_id="c",
            prompt_token_ids=(10, 11, 12),
            phase=ExecutionPhase.PREFILL,
            slot_start=20,
        ),
    )

    layout = BatchLayout(executions=executions)

    assert (
        layout.query_offsets[-1] + layout.query_lengths[-1] == layout.total_query_tokens
    )

    assert layout.key_offsets[-1] + layout.key_lengths[-1] == layout.total_key_tokens

    assert layout.total_query_tokens == sum(layout.query_lengths)
    assert layout.total_key_tokens == sum(layout.key_lengths)

    assert len(layout.query_lengths) == len(executions)
    assert len(layout.key_lengths) == len(executions)
    assert len(layout.query_offsets) == len(executions)
    assert len(layout.key_offsets) == len(executions)
    assert len(layout.logit_indices) == len(executions)
