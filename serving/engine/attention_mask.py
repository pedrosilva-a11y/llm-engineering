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


def semantic_to_additive_attention_mask(
    semantic_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a semantic attention mask to Hugging Face additive format.

    The engine semantic mask uses ``True`` for allowed attention and ``False`` for
    blocked attention. Hugging Face eager attention expects a floating-point additive
    mask shaped ``(batch, heads, query_tokens, key_tokens)``.

    Args:
        semantic_mask: Boolean mask with shape
            ``(total_query_tokens, total_key_tokens)``.
        dtype: Floating-point dtype used by the model attention computation.

    Returns:
        Additive attention mask with shape
            ``(1, 1, total_query_tokens, total_key_tokens)``.

    Raises:
        ValueError: If the semantic mask geometry or dtype is unsupported.
    """
    if semantic_mask.ndim != 2:
        raise ValueError("Semantic attention mask must be two-dimensional.")

    if semantic_mask.dtype != torch.bool:
        raise ValueError("Semantic attention mask must use boolean dtype.")

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Additive attention mask must use a floating-point dtype.")

    additive_mask = torch.zeros(
        semantic_mask.shape,
        dtype=dtype,
        device=semantic_mask.device,
    )

    additive_mask.masked_fill_(~semantic_mask, torch.finfo(dtype).min)

    return additive_mask.unsqueeze(0).unsqueeze(0)
