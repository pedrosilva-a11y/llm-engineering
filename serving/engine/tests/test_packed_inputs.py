"""Tests for packed model input construction."""

import torch

from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase
from serving.engine.packed_inputs import build_packed_inputs
from serving.engine.tests.helpers import make_execution


def test_single_prefill_packs_full_history_and_local_positions() -> None:
    """Pack the full sequence and zero-based positions for prefill."""
    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11, 12),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))
    packed = build_packed_inputs(layout, device=torch.device("cpu"))

    assert torch.equal(packed.input_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    assert torch.equal(packed.position_ids, torch.tensor([0, 1, 2], dtype=torch.long))


def test_single_decode_packs_only_newest_token_at_logical_position() -> None:
    """Pack only the newest token while preserving its sequence-local position."""
    execution = make_execution(
        request_id="decode",
        prompt_token_ids=(20, 21, 22),
        generated_token_ids=(23,),
        phase=ExecutionPhase.DECODE,
    )

    layout = BatchLayout(executions=(execution,))
    packed = build_packed_inputs(layout, device=torch.device("cpu"))

    assert torch.equal(packed.input_ids, torch.tensor([23], dtype=torch.long))
    assert torch.equal(packed.position_ids, torch.tensor([3], dtype=torch.long))


def test_mixed_batch_packs_queries_inmake_execution_order() -> None:
    """Pack mixed prefill and decode queries using sequence-local positions."""
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
    packed = build_packed_inputs(layout, device=torch.device("cpu"))

    assert torch.equal(
        packed.input_ids, torch.tensor([10, 11, 12, 23, 30, 31], dtype=torch.long)
    )

    assert torch.equal(
        packed.position_ids, torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    )

    assert packed.input_ids.numel() == layout.total_query_tokens
    assert packed.position_ids.numel() == layout.total_query_tokens


def test_all_decode_batch_preserves_each_sequence_logical_position() -> None:
    """Use independent logical positions for requests in a decode-only batch."""
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
    packed = build_packed_inputs(layout, device=torch.device("cpu"))

    assert torch.equal(packed.input_ids, torch.tensor([14, 27, 32], dtype=torch.long))
    assert torch.equal(packed.position_ids, torch.tensor([4, 7, 2], dtype=torch.long))


def test_reprefill_packs_prompt_and_generated_history() -> None:
    """Recompute the complete logical history during re-prefill."""
    execution = make_execution(
        request_id="reprefill",
        prompt_token_ids=(10, 11, 12),
        generated_token_ids=(40, 41),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))
    packed = build_packed_inputs(layout, device=torch.device("cpu"))

    assert torch.equal(
        packed.input_ids, torch.tensor([10, 11, 12, 40, 41], dtype=torch.long)
    )

    assert torch.equal(
        packed.position_ids, torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    )


def test_packed_inputs_use_long_dtype_and_requested_device() -> None:
    """Create model input tensors with integer dtype on the requested device."""
    execution = make_execution(
        request_id="prefill",
        prompt_token_ids=(10, 11),
        phase=ExecutionPhase.PREFILL,
    )

    layout = BatchLayout(executions=(execution,))
    device = torch.device("cpu")

    packed = build_packed_inputs(layout, device=device)

    assert packed.input_ids.dtype is torch.long
    assert packed.position_ids.dtype is torch.long

    assert packed.input_ids.device == device
    assert packed.position_ids.device == device
