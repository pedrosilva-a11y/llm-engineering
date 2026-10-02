"""Shared test helpers for inference-engine execution tests."""

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.request import Request
from serving.engine.sequence import SequenceState


def make_execution(
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    phase: ExecutionPhase,
    generated_token_ids: tuple[int, ...] = (),
    slot_start: int = 0,
) -> ModelExecution:
    """Build a model execution with deterministic physical slots.

    Args:
        request_id: Identifier for the synthetic request.
        prompt_token_ids: Prompt token identifiers for the sequence.
        phase: Execution phase represented by the model execution.
        generated_token_ids: Tokens already generated for the sequence.
        slot_start: First physical slot identifier assigned to the sequence.

    Returns:
        Model execution with slot identifiers covering the complete current logical
        sequence history.
    """
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=16,
    )

    sequence = SequenceState(
        request=request,
        generated_token_ids=list(generated_token_ids),
    )

    slot_ids = tuple(range(slot_start, slot_start + sequence.current_length))

    return ModelExecution(
        sequence=sequence,
        phase=phase,
        slot_ids=slot_ids,
    )
