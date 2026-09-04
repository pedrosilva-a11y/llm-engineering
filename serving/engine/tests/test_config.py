"""Tests for LLM inference engine configuration."""

import pytest

from serving.engine.config import EngineConfiguration


def test_engine_configuration_defaults() -> None:
    """Create engine configuration with expected default values."""
    configuration = EngineConfiguration()

    assert configuration.device == "cpu"
    assert configuration.max_sequences == 8
    assert configuration.eos_token_id == 0


def test_engine_configuration_explicit_values() -> None:
    """Create engine configuration with explicitly provided values."""
    configuration = EngineConfiguration(
        device="cuda",
        max_sequences=16,
        eos_token_id=2,
    )

    assert configuration.device == "cuda"
    assert configuration.max_sequences == 16
    assert configuration.eos_token_id == 2


@pytest.mark.parametrize(
    "device",
    [
        "",
        "   ",
    ],
)
def test_empty_device_is_rejected(
    device: str,
) -> None:
    """Reject empty or whitespace-only device names."""
    with pytest.raises(ValueError, match="device must not be empty"):
        EngineConfiguration(device=device)


@pytest.mark.parametrize(
    "max_sequences",
    [
        0,
        -1,
    ],
)
def test_non_positive_max_sequences_is_rejected(
    max_sequences: int,
) -> None:
    """Reject non-positive maximum sequence counts."""
    with pytest.raises(ValueError, match="max_sequences must be greater than zero"):
        EngineConfiguration(max_sequences=max_sequences)


@pytest.mark.parametrize(
    "eos_token_id",
    [
        -1,
        -10,
    ],
)
def test_negative_eos_token_id_is_rejected(
    eos_token_id: int,
) -> None:
    """Reject negative end-of-sequence token identifiers."""
    with pytest.raises(ValueError, match="eos_token_id must not be negative"):
        EngineConfiguration(eos_token_id=eos_token_id)
