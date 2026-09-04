"""Tests for LLM inference engine model runners."""

import pytest
import torch

from serving.engine.model_runner import (
    DeterministicStubModelRunner,
    ModelRunner,
)
from serving.engine.request import Request
from serving.engine.sequence import SequenceState


def test_deterministic_stub_model_runner_forward() -> None:
    """Produce deterministic next-token logits for a sequence batch."""
    runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        device="cpu",
    )

    sequences = [
        SequenceState(
            request=Request(
                request_id="request-1",
                prompt_token_ids=(1, 2, 3),
                max_new_tokens=4,
            ),
        ),
        SequenceState(
            request=Request(
                request_id="request-2",
                prompt_token_ids=(4, 5, 6),
                max_new_tokens=4,
            ),
        ),
    ]

    logits = runner.forward(sequences)

    assert logits.shape == (2, 10)
    assert logits.device == torch.device("cpu")
    assert torch.argmax(logits, dim=1).tolist() == [5, 5]


def test_deterministic_stub_model_runner_eos_thresholds() -> None:
    """Emit EOS after each request reaches its configured token threshold."""
    runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        eos_after_by_request={
            "request-1": 1,
        },
        default_eos_after=2,
    )

    first_sequence = SequenceState(
        request=Request(
            request_id="request-1",
            prompt_token_ids=(1, 2),
            max_new_tokens=4,
        ),
        generated_token_ids=[5],
    )
    second_sequence = SequenceState(
        request=Request(
            request_id="request-2",
            prompt_token_ids=(3, 4),
            max_new_tokens=4,
        ),
        generated_token_ids=[5],
    )
    third_sequence = SequenceState(
        request=Request(
            request_id="request-3",
            prompt_token_ids=(5, 6),
            max_new_tokens=4,
        ),
        generated_token_ids=[5, 5],
    )

    logits = runner.forward(
        [
            first_sequence,
            second_sequence,
            third_sequence,
        ]
    )

    assert torch.argmax(logits, dim=1).tolist() == [0, 5, 0]


def test_deterministic_stub_satisfies_model_runner_protocol() -> None:
    """Keep the deterministic stub compatible with the ModelRunner contract."""
    runner: ModelRunner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
    )

    assert callable(runner.forward)


@pytest.mark.parametrize(
    "vocab_size",
    [
        0,
        -1,
    ],
)
def test_non_positive_vocab_size_is_rejected(
    vocab_size: int,
) -> None:
    """Reject non-positive vocabulary sizes."""
    with pytest.raises(ValueError, match="vocab_size must be greater than zero"):
        DeterministicStubModelRunner(
            vocab_size=vocab_size,
            eos_token_id=0,
            generated_token_id=5,
        )


@pytest.mark.parametrize(
    "eos_token_id",
    [
        -1,
        10,
    ],
)
def test_out_of_vocabulary_eos_token_id_is_rejected(
    eos_token_id: int,
) -> None:
    """Reject EOS token identifiers outside the vocabulary."""
    with pytest.raises(ValueError, match="eos_token_id must be within the vocabulary"):
        DeterministicStubModelRunner(
            vocab_size=10,
            eos_token_id=eos_token_id,
            generated_token_id=5,
        )


@pytest.mark.parametrize(
    "generated_token_id",
    [
        -1,
        10,
    ],
)
def test_out_of_vocabulary_generated_token_id_is_rejected(
    generated_token_id: int,
) -> None:
    """Reject generated token identifiers outside the vocabulary."""
    with pytest.raises(
        ValueError,
        match="generated_token_id must be within the vocabulary",
    ):
        DeterministicStubModelRunner(
            vocab_size=10,
            eos_token_id=0,
            generated_token_id=generated_token_id,
        )


def test_matching_generated_and_eos_token_ids_are_rejected() -> None:
    """Require generated and EOS token identifiers to be different."""
    with pytest.raises(
        ValueError,
        match="generated_token_id must be different from eos_token_id",
    ):
        DeterministicStubModelRunner(
            vocab_size=10,
            eos_token_id=5,
            generated_token_id=5,
        )


@pytest.mark.parametrize(
    "default_eos_after",
    [
        0,
        -1,
    ],
)
def test_non_positive_default_eos_threshold_is_rejected(
    default_eos_after: int,
) -> None:
    """Reject non-positive default EOS thresholds."""
    with pytest.raises(
        ValueError,
        match="default_eos_after must be greater than zero",
    ):
        DeterministicStubModelRunner(
            vocab_size=10,
            eos_token_id=0,
            generated_token_id=5,
            default_eos_after=default_eos_after,
        )


@pytest.mark.parametrize(
    "eos_after",
    [
        0,
        -1,
    ],
)
def test_non_positive_request_eos_threshold_is_rejected(
    eos_after: int,
) -> None:
    """Reject non-positive request-specific EOS thresholds."""
    with pytest.raises(ValueError, match="EOS thresholds must be greater than zero"):
        DeterministicStubModelRunner(
            vocab_size=10,
            eos_token_id=0,
            generated_token_id=5,
            eos_after_by_request={"request-1": eos_after},
        )


def test_empty_sequence_batch_is_rejected() -> None:
    """Reject model execution without active sequences."""
    runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
    )

    with pytest.raises(ValueError, match="sequences must not be empty"):
        runner.forward([])
