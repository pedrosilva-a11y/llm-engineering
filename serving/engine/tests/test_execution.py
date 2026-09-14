"""Tests for model execution metadata."""

from dataclasses import FrozenInstanceError

import pytest

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.request import Request
from serving.engine.sequence import SequenceState


def make_sequence(
    prompt_token_ids: tuple[int, ...] = (1, 2, 3),
    generated_token_ids: tuple[int, ...] = (),
) -> SequenceState:
    """Create a sequence with optional generated history."""
    sequence = SequenceState(
        request=Request(
            request_id="request-1",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=8,
        ),
    )
    sequence.generated_token_ids.extend(generated_token_ids)

    return sequence


def test_prefill_execution_stores_complete_sequence_slots() -> None:
    """Represent every logical position during prefill."""
    sequence = make_sequence(prompt_token_ids=(10, 20, 30))

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(16, 17, 18),
    )

    assert execution.sequence is sequence
    assert execution.phase is ExecutionPhase.PREFILL
    assert execution.slot_ids == (16, 17, 18)
    assert len(execution.slot_ids) == sequence.current_length


def test_decode_execution_stores_complete_sequence_slots() -> None:
    """Preserve the full KV-history mapping during decode."""
    sequence = make_sequence(
        prompt_token_ids=(10, 20, 30),
        generated_token_ids=(40, 50),
    )

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(16, 17, 18, 48, 49),
    )

    assert execution.phase is ExecutionPhase.DECODE
    assert execution.slot_ids == (16, 17, 18, 48, 49)
    assert execution.slot_ids[-1] == 49
    assert len(execution.slot_ids) == sequence.current_length


def test_prefill_can_include_generated_history() -> None:
    """Support re-prefill after preemption without inferring phase from history."""
    sequence = make_sequence(
        prompt_token_ids=(10, 20, 30),
        generated_token_ids=(40, 50),
    )

    assert sequence.num_generated_tokens > 0

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(0, 1, 2, 3, 4),
    )

    assert execution.phase is ExecutionPhase.PREFILL
    assert len(execution.slot_ids) == sequence.current_length


def test_execution_rejects_missing_sequence_slots() -> None:
    """Reject metadata that does not map every logical token position."""
    sequence = make_sequence(
        prompt_token_ids=(10, 20, 30),
    )

    with pytest.raises(
        ValueError,
        match="slot_ids length must equal the sequence current length",
    ):
        ModelExecution(
            sequence=sequence,
            phase=ExecutionPhase.PREFILL,
            slot_ids=(0, 1),
        )


def test_execution_rejects_extra_sequence_slots() -> None:
    """Reject slot mappings longer than the current sequence."""
    sequence = make_sequence(
        prompt_token_ids=(10, 20, 30),
    )

    with pytest.raises(
        ValueError,
        match="slot_ids length must equal the sequence current length",
    ):
        ModelExecution(
            sequence=sequence,
            phase=ExecutionPhase.PREFILL,
            slot_ids=(0, 1, 2, 3),
        )


def test_execution_rejects_negative_slot_id() -> None:
    """Reject invalid physical KV-cache slot identifiers."""
    sequence = make_sequence(
        prompt_token_ids=(10, 20, 30),
    )

    with pytest.raises(
        ValueError,
        match="slot_ids must not contain negative values",
    ):
        ModelExecution(
            sequence=sequence,
            phase=ExecutionPhase.PREFILL,
            slot_ids=(0, -1, 2),
        )


def test_execution_is_immutable() -> None:
    """Prevent execution metadata from changing after construction."""
    sequence = make_sequence()

    execution = ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(0, 1, 2),
    )

    with pytest.raises(FrozenInstanceError):
        setattr(
            execution,
            "slot_ids",
            (3, 4, 5),
        )
