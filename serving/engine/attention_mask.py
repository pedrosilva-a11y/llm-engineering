"""Semantic attention masks for packed model execution."""

import torch

from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase


def build_attention_mask(layout: BatchLayout, *, device: torch.device) -> torch.Tensor:
    """Build a block-diagonal causal attention mask for packed executions.

    The returned mask represents packed query-by-key attention semantics. Each execution
    may attend only to keys belonging to the same sequence.

    Prefill executions use causal attention within their own key region, so each query
    may attend to its current position and all preceding positions. Decode executions
    contribute one query that may attend to the complete current key history, including
    the newest token's own key position.

    The current prefill representation assumes that the query region spans the complete
    key history. Future partial or chunked prefill must account for the query block's
    logical offset within a longer key history.

    The mask is intentionally model-agnostic. It remains two-dimensional and boolean
    so Hugging Face or model-specific mask formatting can be handled at the model
    boundary.

    Args:
        layout: Packed batch geometry describing query and key regions.
        device: Torch device on which the mask should be created.

    Returns:
        Boolean tensor with shape
        ``(layout.total_query_tokens, layout.total_key_tokens)``. ``True`` means the
        query may attend to the corresponding key position.
    """
    mask = torch.zeros(
        (
            layout.total_query_tokens,
            layout.total_key_tokens,
        ),
        dtype=torch.bool,
        device=device,
    )

    for index, execution in enumerate(layout.executions):
        query_offset = layout.query_offsets[index]
        key_offset = layout.key_offsets[index]

        query_length = layout.query_lengths[index]
        key_length = layout.key_lengths[index]

        if execution.phase is ExecutionPhase.PREFILL:
            local_mask = torch.tril(
                torch.ones(
                    (query_length, key_length),
                    dtype=torch.bool,
                    device=device,
                )
            )

        elif execution.phase is ExecutionPhase.DECODE:
            local_mask = torch.ones(
                (query_length, key_length),
                dtype=torch.bool,
                device=device,
            )

        else:
            raise ValueError(f"Unsupported execution phase: {execution.phase}")

        mask[
            query_offset : query_offset + query_length,
            key_offset : key_offset + key_length,
        ] = local_mask

    return mask
