"""Tests for packed block-diagonal causal attention masks."""

import pytest
import torch

from serving.engine.attention_mask import (
    build_attention_mask,
    semantic_to_additive_attention_mask,
)
from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase
from serving.engine.tests.helpers import make_execution

# Attention mask building


def test_single_prefill_builds_lower_triangular_mask() -> None:
    """Allow each prefill query to attend only to itself and earlier keys."""
    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11, 12),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))
    mask = build_attention_mask(layout, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)


def test_unequal_prefills_build_independent_causal_blocks() -> None:
    """Keep unequal prefill sequences isolated in independent causal blocks."""
    executions = (
        make_execution(
            request_id="a",
            prompt_token_ids=(10, 11),
            phase=ExecutionPhase.PREFILL,
            slot_start=0,
        ),
        make_execution(
            request_id="b",
            prompt_token_ids=(20, 21, 22, 23),
            phase=ExecutionPhase.PREFILL,
            slot_start=10,
        ),
    )

    layout = BatchLayout(executions=executions)
    mask = build_attention_mask(layout, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, True, True, False, False],
            [False, False, True, True, True, False],
            [False, False, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)


def test_single_decode_attends_to_complete_current_history() -> None:
    """Allow a decode query to attend to every key in its current history."""
    execution = make_execution(
        request_id="decode",
        prompt_token_ids=(20, 21, 22),
        generated_token_ids=(23,),
        phase=ExecutionPhase.DECODE,
    )

    layout = BatchLayout(executions=(execution,))
    mask = build_attention_mask(layout, device=torch.device("cpu"))

    expected = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    assert torch.equal(mask, expected)


def test_all_decode_batch_keeps_key_histories_isolated() -> None:
    """Give each decode query access only to its own complete key history."""
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
    mask = build_attention_mask(layout, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)


def test_mixed_batch_builds_block_diagonal_causal_mask() -> None:
    """Combine independent prefill and decode attention regions."""
    executions = (
        make_execution(
            request_id="a",
            prompt_token_ids=(10, 11, 12),
            phase=ExecutionPhase.PREFILL,
            slot_start=0,
        ),
        make_execution(
            request_id="b",
            prompt_token_ids=(20, 21, 22),
            generated_token_ids=(23,),
            phase=ExecutionPhase.DECODE,
            slot_start=10,
        ),
        make_execution(
            request_id="c",
            prompt_token_ids=(30, 31),
            phase=ExecutionPhase.PREFILL,
            slot_start=20,
        ),
    )

    layout = BatchLayout(executions=executions)
    mask = build_attention_mask(layout, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [True, False, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False, False],
            [False, False, False, True, True, True, True, False, False],
            [False, False, False, False, False, False, False, True, False],
            [False, False, False, False, False, False, False, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)


def test_reprefill_builds_causal_mask_over_complete_history() -> None:
    """Apply ordinary causal attention to prompt plus generated history."""
    execution = make_execution(
        request_id="reprefill",
        prompt_token_ids=(10, 11, 12),
        generated_token_ids=(40, 41),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))
    mask = build_attention_mask(layout, device=torch.device("cpu"))

    expected = torch.tril(torch.ones((5, 5), dtype=torch.bool))

    assert torch.equal(mask, expected)


def test_attention_mask_matches_layout_shape_dtype_and_device() -> None:
    """Match packed query-key geometry and requested tensor properties."""
    executions = (
        make_execution(
            request_id="a",
            prompt_token_ids=(10, 11, 12),
            phase=ExecutionPhase.PREFILL,
        ),
        make_execution(
            request_id="b",
            prompt_token_ids=(20, 21, 22),
            generated_token_ids=(23,),
            phase=ExecutionPhase.DECODE,
            slot_start=10,
        ),
    )

    layout = BatchLayout(executions=executions)

    device = torch.device("cpu")
    mask = build_attention_mask(layout, device=device)

    assert mask.shape == (layout.total_query_tokens, layout.total_key_tokens)
    assert mask.dtype == torch.bool
    assert mask.device == device


# Semantic to additive attention mask


def test_additive_mask_converts_allowed_and_blocked_positions() -> None:
    """Convert allowed positions to zero and blocked positions to finite minimum."""
    semantic_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )

    mask = semantic_to_additive_attention_mask(semantic_mask, dtype=torch.float32)

    blocked = torch.finfo(torch.float32).min

    expected = torch.tensor(
        [
            [
                [
                    [0.0, blocked, blocked],
                    [0.0, 0.0, blocked],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(mask, expected)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
)
def test_additive_mask_matches_model_facing_shape_dtype_and_device(
    dtype: torch.dtype,
) -> None:
    """Produce model-facing mask geometry with the requested floating dtype."""
    semantic_mask = torch.tensor(
        [
            [True, False],
            [True, True],
        ],
        dtype=torch.bool,
    )

    mask = semantic_to_additive_attention_mask(semantic_mask, dtype=dtype)

    assert mask.shape == (1, 1, 2, 2)
    assert mask.dtype == dtype
    assert mask.device == semantic_mask.device


def test_additive_mask_rejects_non_two_dimensional_semantic_mask() -> None:
    """Reject semantic masks outside query-by-key geometry."""
    semantic_mask = torch.ones((1, 2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="must be two-dimensional"):
        semantic_to_additive_attention_mask(semantic_mask, dtype=torch.float32)


def test_additive_mask_rejects_non_boolean_semantic_mask() -> None:
    """Require semantic masks to use the engine's boolean representation."""
    semantic_mask = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="must use boolean dtype"):
        semantic_to_additive_attention_mask(semantic_mask, dtype=torch.float32)


def test_additive_mask_rejects_non_floating_output_dtype() -> None:
    """Require the model-facing additive mask to use floating-point values."""
    semantic_mask = torch.ones((2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="must use a floating-point dtype"):
        semantic_to_additive_attention_mask(semantic_mask, dtype=torch.int64)
