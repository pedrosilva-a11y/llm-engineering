"""Tests for the naive reference model runner."""

from typing import Any, cast

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedModel

from serving.engine.naive_model_runner import NaiveModelRunner
from serving.engine.request import Request
from serving.engine.sequence import SequenceState


@pytest.fixture
def tiny_model() -> PreTrainedModel:
    """Return a tiny randomly initialized causal language model."""
    torch.manual_seed(0)

    configuration = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
    )

    model_class: Any = GPT2LMHeadModel
    return cast(PreTrainedModel, model_class(configuration))


def make_sequence(
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: list[int] | None = None,
) -> SequenceState:
    """Create a sequence state for model-runner tests."""
    sequence = SequenceState(
        request=Request(
            request_id="request",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=8,
        ),
    )

    if generated_token_ids is not None:
        sequence.generated_token_ids.extend(generated_token_ids)

    return sequence


def test_forward_returns_expected_shape(tiny_model: PreTrainedModel) -> None:
    """Return one next-token logit vector per sequence."""
    runner = NaiveModelRunner(model=tiny_model, device=torch.device("cpu"))

    sequences = (make_sequence((1, 4, 5)), make_sequence((1, 6, 7, 8)))

    logits = runner.forward(sequences)

    assert logits.shape == (2, 32)


def test_forward_uses_complete_sequence_history(tiny_model: PreTrainedModel) -> None:
    """Use prompt and previously generated tokens for inference."""
    runner = NaiveModelRunner(model=tiny_model, device=torch.device("cpu"))

    sequence = make_sequence(prompt_token_ids=(1, 4, 5), generated_token_ids=[6, 7])

    actual = runner.forward((sequence,))

    input_ids = torch.tensor(sequence.all_token_ids, dtype=torch.long).unsqueeze(0)

    model_callable: Any = tiny_model

    with torch.inference_mode():
        outputs: Any = model_callable(input_ids=input_ids, use_cache=False)

    expected = cast(torch.Tensor, outputs.logits)[0, -1, :]

    torch.testing.assert_close(actual[0], expected)


def test_forward_supports_different_sequence_lengths(
    tiny_model: PreTrainedModel,
) -> None:
    """Evaluate variable-length sequences independently."""
    runner = NaiveModelRunner(model=tiny_model, device=torch.device("cpu"))

    short_sequence = make_sequence((1, 3))
    long_sequence = make_sequence((1, 3, 4, 5, 6, 7))

    logits = runner.forward((short_sequence, long_sequence))

    assert logits.shape == (2, 32)


def test_forward_returns_logits_on_configured_device(
    tiny_model: PreTrainedModel,
) -> None:
    """Return logits on the configured inference device."""
    runner = NaiveModelRunner(model=tiny_model, device=torch.device("cpu"))

    logits = runner.forward(
        (make_sequence((1, 4, 5)),),
    )

    assert logits.device.type == "cpu"


def test_constructor_sets_model_to_evaluation_mode(
    tiny_model: PreTrainedModel,
) -> None:
    """Put the underlying model into evaluation mode."""
    model_module = cast(torch.nn.Module, tiny_model)
    model_module.train()

    NaiveModelRunner(model=tiny_model, device=torch.device("cpu"))

    assert not model_module.training


def test_forward_rejects_empty_batch(tiny_model: PreTrainedModel) -> None:
    """Reject inference without any sequences."""
    runner = NaiveModelRunner(model=tiny_model, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="At least one sequence is required."):
        runner.forward(())
