"""Tests for LLM inference engine sequence state."""

import pytest

from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceState, SequenceStatus


@pytest.fixture
def generation_request() -> Request:
    """Create a valid generation request for sequence-state tests."""
    return Request(
        request_id="request-1",
        prompt_token_ids=(10, 20, 30),
        max_new_tokens=4,
    )


def test_sequence_state_defaults(
    generation_request: Request,
) -> None:
    """Create a sequence with the expected initial execution state."""
    sequence = SequenceState(request=generation_request)

    assert sequence.generated_token_ids == []
    assert sequence.status == SequenceStatus.WAITING
    assert sequence.finish_reason is None
    assert sequence.is_finished is False


def test_sequence_token_properties(
    generation_request: Request,
) -> None:
    """Calculate generated-token count, total length, and complete token ids."""
    sequence = SequenceState(
        request=generation_request,
        generated_token_ids=[40, 50],
    )

    assert sequence.num_generated_tokens == 2
    assert sequence.current_length == 5
    assert sequence.all_token_ids == (10, 20, 30, 40, 50)


def test_generated_token_lists_are_independent(
    generation_request: Request,
) -> None:
    """Keep generated-token storage independent across sequence instances."""
    first_sequence = SequenceState(request=generation_request)
    second_sequence = SequenceState(request=generation_request)

    first_sequence.append_token(40)

    assert first_sequence.generated_token_ids == [40]
    assert second_sequence.generated_token_ids == []


def test_append_token(
    generation_request: Request,
) -> None:
    """Append one generated token and update derived sequence properties."""
    sequence = SequenceState(request=generation_request)

    sequence.append_token(40)

    assert sequence.generated_token_ids == [40]
    assert sequence.num_generated_tokens == 1
    assert sequence.current_length == 4
    assert sequence.all_token_ids == (10, 20, 30, 40)


def test_mark_running(
    generation_request: Request,
) -> None:
    """Transition a waiting sequence into the running state."""
    sequence = SequenceState(request=generation_request)

    sequence.mark_running()

    assert sequence.status == SequenceStatus.RUNNING
    assert sequence.finish_reason is None
    assert sequence.is_finished is False


def test_mark_preempted_preserves_generation_state(
    generation_request: Request,
) -> None:
    """Preempt a running sequence without losing generated-token progress."""
    sequence = SequenceState(request=generation_request)

    sequence.mark_running()
    sequence.append_token(40)
    sequence.append_token(50)

    sequence.mark_preempted()

    assert sequence.status == SequenceStatus.PREEMPTED
    assert sequence.generated_token_ids == [40, 50]
    assert sequence.num_generated_tokens == 2
    assert sequence.current_length == 5

    assert sequence.finish_reason is None
    assert sequence.is_finished is False


def test_preempted_sequence_can_be_marked_running_again(
    generation_request: Request,
) -> None:
    """Return a preempted sequence to the running state."""
    sequence = SequenceState(request=generation_request)

    sequence.mark_running()
    sequence.mark_preempted()
    sequence.mark_running()

    assert sequence.status == SequenceStatus.RUNNING
    assert sequence.finish_reason is None
    assert sequence.is_finished is False


def test_waiting_sequence_cannot_be_marked_preempted(
    generation_request: Request,
) -> None:
    """Reject preemption of a sequence that is not running."""
    sequence = SequenceState(request=generation_request)

    with pytest.raises(RuntimeError, match="Only a running sequence can be preempted"):
        sequence.mark_preempted()

    assert sequence.status == SequenceStatus.WAITING


@pytest.mark.parametrize(
    "finish_reason",
    [
        FinishReason.EOS,
        FinishReason.LENGTH,
        FinishReason.CANCELLED,
    ],
)
def test_mark_finished(
    generation_request: Request,
    finish_reason: FinishReason,
) -> None:
    """Transition a sequence into a consistent terminal state."""
    sequence = SequenceState(request=generation_request)

    sequence.mark_finished(reason=finish_reason)

    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.finish_reason == finish_reason
    assert sequence.is_finished is True


def test_append_token_to_finished_sequence_is_rejected(
    generation_request: Request,
) -> None:
    """Reject generated tokens after sequence completion."""
    sequence = SequenceState(request=generation_request)
    sequence.mark_finished(reason=FinishReason.EOS)

    with pytest.raises(
        ValueError,
        match="Cannot append a token to a finished sequence",
    ):
        sequence.append_token(40)


def test_mark_finished_sequence_as_running_is_rejected(
    generation_request: Request,
) -> None:
    """Reject transitions from a finished sequence back to running."""
    sequence = SequenceState(request=generation_request)
    sequence.mark_finished(reason=FinishReason.EOS)

    with pytest.raises(
        RuntimeError,
        match="Only a waiting or preempted sequence can be marked running",
    ):
        sequence.mark_running()


def test_finished_sequence_cannot_be_finished_again(
    generation_request: Request,
) -> None:
    """Reject repeated attempts to finish an already completed sequence."""
    sequence = SequenceState(request=generation_request)
    sequence.mark_finished(reason=FinishReason.EOS)

    with pytest.raises(
        ValueError,
        match="Cannot finish an already finished sequence",
    ):
        sequence.mark_finished(reason=FinishReason.LENGTH)
