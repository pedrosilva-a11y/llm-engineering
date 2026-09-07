"""Tests for the LLM inference engine."""

from unittest.mock import patch

import pytest

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceStatus


@pytest.fixture
def configuration() -> EngineConfiguration:
    """Create a small engine configuration for control-flow tests."""
    return EngineConfiguration(
        device="cpu",
        max_sequences=2,
        eos_token_id=0,
    )


@pytest.fixture
def model_runner() -> DeterministicStubModelRunner:
    """Create a deterministic model runner for engine tests."""
    return DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        eos_after_by_request={
            "request-1": 2,
            "request-2": 4,
            "request-3": 1,
        },
        device="cpu",
    )


@pytest.fixture
def engine(
    configuration: EngineConfiguration,
    model_runner: DeterministicStubModelRunner,
) -> Engine:
    """Create an inference engine for control-flow tests."""
    return Engine(
        configuration=configuration,
        model_runner=model_runner,
    )


def test_submit_keeps_requests_waiting_until_scheduled(
    engine: Engine,
) -> None:
    """Keep submitted requests waiting until an engine step schedules them."""
    max_new_tokens = 8
    request_id1 = "request-1"
    request_id2 = "request-2"
    request_id3 = "request-3"

    requests = [
        Request(
            request_id=request_id1,
            prompt_token_ids=(1, 2),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id=request_id2,
            prompt_token_ids=(3, 4),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id=request_id3,
            prompt_token_ids=(5, 6),
            max_new_tokens=max_new_tokens,
        ),
    ]

    for request in requests:
        engine.submit(request)

    assert engine.running_sequences == ()

    assert [sequence.request.request_id for sequence in engine.waiting_sequences] == [
        request_id1,
        request_id2,
        request_id3,
    ]

    assert all(
        sequence.status == SequenceStatus.WAITING
        for sequence in engine.waiting_sequences
    )

    assert engine.finished_sequences == ()
    assert engine.has_unfinished_requests is True


def test_duplicate_request_id_is_rejected(
    engine: Engine,
) -> None:
    """Reject a request identifier that has already been submitted."""
    max_new_tokens = 8

    first_request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2),
        max_new_tokens=max_new_tokens,
    )

    engine.submit(first_request)

    duplicate_request = Request(
        request_id="request-1",
        prompt_token_ids=(3, 4),
        max_new_tokens=max_new_tokens,
    )

    with pytest.raises(ValueError, match="has already been submitted"):
        engine.submit(duplicate_request)


def test_step_generates_one_token_per_running_sequence(
    engine: Engine,
) -> None:
    """Schedule requests and generate one token for each selected sequence."""
    max_new_tokens = 8

    requests = [
        Request(
            request_id="request-1",
            prompt_token_ids=(1, 2),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id="request-2",
            prompt_token_ids=(3, 4),
            max_new_tokens=max_new_tokens,
        ),
    ]

    for request in requests:
        engine.submit(request)

    assert engine.running_sequences == ()
    assert len(engine.waiting_sequences) == 2

    engine.step()

    assert len(engine.running_sequences) == 2

    assert all(
        sequence.num_generated_tokens == 1 for sequence in engine.running_sequences
    )

    assert engine.waiting_sequences == ()


def test_step_uses_one_batched_model_runner_call(
    configuration: EngineConfiguration,
) -> None:
    """Call the model runner once for a batch of running sequences."""
    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        device="cpu",
    )

    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    max_new_tokens = 8
    requests = [
        Request(
            request_id="request-1",
            prompt_token_ids=(1, 2),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id="request-2",
            prompt_token_ids=(3, 4),
            max_new_tokens=max_new_tokens,
        ),
    ]

    for request in requests:
        engine.submit(request)

    with patch.object(
        model_runner,
        "forward",
        wraps=model_runner.forward,
    ) as forward_mock:
        engine.step()

        forward_mock.assert_called_once()


def test_step_without_active_sequence_skips_model_runner(
    configuration: EngineConfiguration,
) -> None:
    """Skip model execution when no sequences are active."""
    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        device="cpu",
    )

    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    with patch.object(
        model_runner, "forward", wraps=model_runner.forward
    ) as forward_mock:
        finished = engine.step()

        assert finished == ()
        forward_mock.assert_not_called()


def test_step_reserves_kv_for_non_terminal_generated_token() -> None:
    """Grow KV storage before appending a non-terminal generated token."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_sequences=1,
        max_batched_tokens=16,
    )
    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        device="cpu",
    )
    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    request_id = "request-1"

    engine.submit(
        Request(
            request_id=request_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=8,
        ),
    )

    engine.step()

    sequence = engine.running_sequences[0]

    assert sequence.generated_token_ids == [5]
    assert engine.scheduler.block_table.blocks_for_request(request_id) == (0, 1)
    assert engine.scheduler.block_pool.num_allocated_blocks == 2


def test_terminal_token_does_not_require_additional_kv_block() -> None:
    """Finish successfully without reserving KV for a terminal token."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=1,
        max_sequences=1,
        max_batched_tokens=16,
    )
    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        device="cpu",
    )
    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    engine.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=1,
        ),
    )

    finished = engine.step()

    assert len(finished) == 1
    assert finished[0].finish_reason == FinishReason.LENGTH

    assert engine.scheduler.block_pool.num_allocated_blocks == 0
    assert engine.scheduler.block_pool.num_free_blocks == 1


def test_sequence_finishes_on_eos(
    engine: Engine,
) -> None:
    """Finish a sequence when the model runner predicts EOS."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2),
        max_new_tokens=8,
    )

    engine.submit(request)

    engine.step()
    engine.step()
    engine.step()

    assert not engine.running_sequences
    assert len(engine.finished_sequences) == 1

    finished_sequence = engine.finished_sequences[0]

    assert finished_sequence.status == SequenceStatus.FINISHED
    assert finished_sequence.finish_reason == FinishReason.EOS
    assert finished_sequence.generated_token_ids == [5, 5, 0]
    assert finished_sequence.is_finished is True


def test_sequence_finishes_at_max_new_tokens(
    engine: Engine,
) -> None:
    """Finish a sequence when its generation-length limit is reached."""
    request = Request(
        request_id="request-2",
        prompt_token_ids=(1, 2),
        max_new_tokens=2,
    )

    engine.submit(request)

    engine.step()
    engine.step()

    assert len(engine.finished_sequences) == 1

    finished_sequence = engine.finished_sequences[0]

    assert finished_sequence.status == SequenceStatus.FINISHED
    assert finished_sequence.finish_reason == FinishReason.LENGTH
    assert finished_sequence.num_generated_tokens == 2
    assert finished_sequence.generated_token_ids == [5, 5]
    assert 0 not in finished_sequence.generated_token_ids


def test_eos_takes_precedence_at_generation_limit(
    configuration: EngineConfiguration,
) -> None:
    """Prefer EOS when it is generated exactly at the length boundary."""
    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        default_eos_after=2,
        device="cpu",
    )

    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    request = Request(
        request_id="boundary-request",
        prompt_token_ids=(1, 2),
        max_new_tokens=3,
    )

    engine.submit(request)

    finished_sequences = engine.run_until_complete()

    assert len(finished_sequences) == 1

    finished_sequence = finished_sequences[0]

    assert finished_sequence.generated_token_ids == [5, 5, 0]
    assert finished_sequence.num_generated_tokens == 3
    assert finished_sequence.finish_reason == FinishReason.EOS


def test_finished_sequence_frees_execution_capacity_for_waiting_request(
    engine: Engine,
) -> None:
    """Schedule a waiting request after execution capacity is released."""
    max_new_tokens = 8
    request_id1 = "request-1"
    request_id2 = "request-2"
    request_id3 = "request-3"

    requests = [
        Request(
            request_id=request_id1,
            prompt_token_ids=(1, 2),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id=request_id2,
            prompt_token_ids=(3, 4),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id=request_id3,
            prompt_token_ids=(5, 6),
            max_new_tokens=max_new_tokens,
        ),
    ]

    for request in requests:
        engine.submit(request)

    assert engine.running_sequences == ()

    assert [sequence.request.request_id for sequence in engine.waiting_sequences] == [
        request_id1,
        request_id2,
        request_id3,
    ]

    engine.step()
    engine.step()
    engine.step()

    # Mypy retains the earlier empty-tuple narrowing across engine.step().
    running_request_ids = [  # type: ignore[var-annotated]
        sequence.request.request_id for sequence in engine.running_sequences
    ]

    assert running_request_ids == [request_id2]

    assert [sequence.request.request_id for sequence in engine.waiting_sequences] == [
        request_id3,
    ]

    assert [sequence.request.request_id for sequence in engine.finished_sequences] == [
        request_id1,
    ]

    engine.step()

    # Mypy retains the earlier empty-tuple narrowing across engine.step().
    running_request_ids = [  # type: ignore[var-annotated]
        sequence.request.request_id for sequence in engine.running_sequences
    ]

    assert running_request_ids == [request_id2, request_id3]

    assert engine.waiting_sequences == ()

    assert [sequence.request.request_id for sequence in engine.finished_sequences] == [
        request_id1,
    ]


def test_step_returns_only_sequences_finished_during_that_step(
    engine: Engine,
) -> None:
    """Return only sequences that complete in the current decoding step."""
    max_new_tokens = 8

    requests = [
        Request(
            request_id="request-1",
            prompt_token_ids=(1, 2),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id="request-2",
            prompt_token_ids=(3, 4),
            max_new_tokens=max_new_tokens,
        ),
    ]

    for request in requests:
        engine.submit(request)

    assert engine.step() == ()
    assert engine.step() == ()

    finished_this_step = engine.step()

    assert [sequence.request.request_id for sequence in finished_this_step] == [
        "request-1"
    ]

    assert [sequence.request.request_id for sequence in engine.running_sequences] == [
        "request-2"
    ]


def test_run_until_complete_finishes_all_requests(
    engine: Engine,
) -> None:
    """Run all submitted requests through their complete generation lifecycle."""
    max_new_tokens = 8

    requests = [
        Request(
            request_id="request-1",
            prompt_token_ids=(1, 2),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id="request-2",
            prompt_token_ids=(3, 4),
            max_new_tokens=max_new_tokens,
        ),
        Request(
            request_id="request-3",
            prompt_token_ids=(5, 6),
            max_new_tokens=max_new_tokens,
        ),
    ]

    for request in requests:
        engine.submit(request)

    finished_sequences = engine.run_until_complete()

    assert not engine.waiting_sequences
    assert not engine.running_sequences
    assert engine.has_unfinished_requests is False

    assert all(
        sequence.status == SequenceStatus.FINISHED for sequence in finished_sequences
    )

    assert [sequence.request.request_id for sequence in finished_sequences] == [
        "request-1",
        "request-2",
        "request-3",
    ]


def test_run_until_complete_handles_requests_that_cannot_share_kv_capacity() -> None:
    """Complete requests sequentially when KV capacity prevents co-residency."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=2,
        max_sequences=2,
        max_batched_tokens=8,
    )
    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        device="cpu",
    )
    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    for request_id in ("request-1", "request-2"):
        engine.submit(
            Request(
                request_id=request_id,
                prompt_token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
                max_new_tokens=1,
            ),
        )

    finished = engine.run_until_complete()

    assert [sequence.request.request_id for sequence in finished] == [
        "request-1",
        "request-2",
    ]

    assert engine.waiting_sequences == ()
    assert engine.running_sequences == ()
    assert engine.scheduler.block_pool.num_free_blocks == 2
