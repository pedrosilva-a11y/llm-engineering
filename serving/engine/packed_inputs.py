"""Packed token and position tensors for model execution."""

from dataclasses import dataclass

import torch

from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase, ModelExecution


@dataclass(frozen=True)
class PackedInputs:
    """Contain packed query tokens and sequence-local position identifiers.

    The tensors represent the query regions described by ``BatchLayout`` in execution
    order. They remain one-dimensional so this layer describes packed query semantics
    without imposing a model-specific batch shape.

    Attributes:
        input_ids: Packed query token identifiers.
        position_ids: Sequence-local positions corresponding to each query token.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor


def build_packed_inputs(layout: BatchLayout, *, device: torch.device) -> PackedInputs:
    """Build packed query-token and position tensors for a batch layout.

    Prefill executions contribute their complete current token history and positions
    starting from zero. Decode executions contribute only their newest token,
    positioned at the final logical position of the current sequence.

    Positional identifiers remain local to each sequence and therefore do not correspond
    to positions in the packed tensor.

    Args:
        layout: Valid packed batch geometry whose executions should be converted into
            query tensors.
        device: Torch device on which the resulting tensors should be created.

    Returns:
        Packed input and position tensors with one element per query token.

    Raises:
        RuntimeError: If the number of constructed tokens does not match the
            query-token count described by the batch layout.
    """
    input_ids: list[int] = []
    position_ids: list[int] = []

    for execution in layout.executions:
        input_ids.extend(_query_token_ids(execution))
        position_ids.extend(_query_position_ids(execution))

    if len(input_ids) != layout.total_query_tokens:
        raise RuntimeError(
            "Packed input token count does not match BatchLayout query geometry.",
        )

    if len(position_ids) != layout.total_query_tokens:
        raise RuntimeError(
            "Packed position count does not match BatchLayout query geometry.",
        )

    return PackedInputs(
        input_ids=torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device,
        ),
        position_ids=torch.tensor(
            position_ids,
            dtype=torch.long,
            device=device,
        ),
    )


def _query_token_ids(execution: ModelExecution) -> tuple[int, ...]:
    """Return query token identifiers contributed by one execution.

    Prefill contributes the complete current logical sequence because all token
    representations must be recomputed. Decode contributes only the newest token
    because previous key/value states are already available from the KV cache.

    Args:
        execution: Model execution whose query tokens should be selected.

    Returns:
        Token identifiers contributed to packed query space.

    Raises:
        ValueError: If the execution phase is not supported.
    """
    if execution.phase is ExecutionPhase.PREFILL:
        return execution.sequence.all_token_ids

    if execution.phase is ExecutionPhase.DECODE:
        return execution.sequence.all_token_ids[-1:]

    raise ValueError(f"Unsupported execution phase: {execution.phase}")


def _query_position_ids(execution: ModelExecution) -> tuple[int, ...]:
    """Return sequence-local positions contributed by one execution.

    Prefill positions span the complete current sequence from zero through
    ``current_length - 1``. Decode contributes only the logical position of the
    newest token, which is ``current_length - 1``.

    These positions are local to the originating sequence rather than offsets in packed
    query space.

    Args:
        execution: Model execution whose query positions should be derived.

    Returns:
        Sequence-local position identifiers for the execution's query tokens.

    Raises:
        ValueError: If the execution phase is not supported.
    """
    if execution.phase is ExecutionPhase.PREFILL:
        return tuple(range(execution.sequence.current_length))

    if execution.phase is ExecutionPhase.DECODE:
        return (execution.sequence.current_length - 1,)

    raise ValueError(f"Unsupported execution phase: {execution.phase}")
