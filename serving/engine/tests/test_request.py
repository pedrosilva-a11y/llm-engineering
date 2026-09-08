"""Tests for LLM inference engine requests."""

import pytest

from serving.engine.request import Request
from serving.engine.sampling import SamplingParameters


def test_request_success() -> None:
    """Create a generation request with valid values."""
    sampling_parameters = SamplingParameters(
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        greedy=False,
    )

    request = Request(
        request_id="request-1",
        prompt_token_ids=(10, 20, 30),
        max_new_tokens=16,
        arrival_time=1.5,
        sampling_parameters=sampling_parameters,
    )

    assert request.request_id == "request-1"
    assert request.prompt_token_ids == (10, 20, 30)
    assert request.max_new_tokens == 16
    assert request.arrival_time == 1.5
    assert request.sampling_parameters == sampling_parameters


def test_request_default_arrival_time() -> None:
    """Use the default arrival timestamp when one is not provided."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=(10, 20, 30),
        max_new_tokens=16,
    )

    assert request.arrival_time == 0.0


def test_request_defaults_to_greedy_sampling() -> None:
    """Use greedy token selection when sampling parameters are not provided."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=(10, 20, 30),
        max_new_tokens=16,
    )

    assert request.sampling_parameters == SamplingParameters()
    assert request.sampling_parameters.greedy is True


@pytest.mark.parametrize(
    "request_id",
    [
        "",
        "   ",
    ],
)
def test_empty_request_id_is_rejected(
    request_id: str,
) -> None:
    """Reject empty or whitespace-only request identifiers."""
    with pytest.raises(ValueError, match="request_id must not be empty"):
        Request(
            request_id=request_id,
            prompt_token_ids=(10, 20, 30),
            max_new_tokens=16,
        )


def test_empty_prompt_token_ids_are_rejected() -> None:
    """Reject generation requests without prompt tokens."""
    with pytest.raises(ValueError, match="prompt_token_ids must not be empty"):
        Request(
            request_id="request-1",
            prompt_token_ids=(),
            max_new_tokens=16,
        )


@pytest.mark.parametrize(
    "max_new_tokens",
    [
        0,
        -1,
    ],
)
def test_non_positive_max_new_tokens_are_rejected(
    max_new_tokens: int,
) -> None:
    """Reject non-positive maximum generation lengths."""
    with pytest.raises(ValueError, match="max_new_tokens must be greater than zero"):
        Request(
            request_id="request-1",
            prompt_token_ids=(10, 20, 30),
            max_new_tokens=max_new_tokens,
        )
