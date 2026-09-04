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


def test_submit_admits_request_up_to_capacity(
    engine: Engine,
) -> None:
    """Admit submitted requests up to the configured sequence capacity."""
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

    assert [sequence.request.request_id for sequence in engine.running_sequences] == [
        "request-1",
        "request-2",
    ]

    assert [sequence.request.request_id for sequence in engine.waiting_sequences] == [
        "request-3"
    ]

    assert all(
        sequence.status == SequenceStatus.RUNNING
        for sequence in engine.running_sequences
    )

    assert all(
        sequence.status == SequenceStatus.WAITING
        for sequence in engine.waiting_sequences
    )

    assert not engine.finished_sequences
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
    """Generate one token for every running sequence in one engine step."""
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

    assert all(
        sequence.num_generated_tokens == 0 for sequence in engine.running_sequences
    )

    engine.step()

    assert all(
        sequence.num_generated_tokens == 1 for sequence in engine.running_sequences
    )


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


def test_finished_sequence_frees_capacity_for_waiting_request(
    engine: Engine,
) -> None:
    """Admit the oldest waiting request when running capacity is released."""
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

    assert [sequence.request.request_id for sequence in engine.running_sequences] == [
        "request-1",
        "request-2",
    ]
    assert [sequence.request.request_id for sequence in engine.waiting_sequences] == [
        "request-3"
    ]

    engine.step()
    engine.step()
    engine.step()

    assert [sequence.request.request_id for sequence in engine.running_sequences] == [
        "request-2",
        "request-3",
    ]
    assert not engine.waiting_sequences

    assert [sequence.request.request_id for sequence in engine.finished_sequences] == [
        "request-1"
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
