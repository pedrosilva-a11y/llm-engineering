"""Tests for LLM inference engine configuration."""

import pytest

from serving.engine.config import EngineConfiguration


def test_engine_configuration_defaults() -> None:
    """Create engine configuration with expected default values."""
    configuration = EngineConfiguration()

    assert configuration.device == "cpu"
    assert configuration.max_sequences == 8
    assert configuration.eos_token_id == 0
    assert configuration.block_size == 16
    assert configuration.num_blocks == 256
    assert configuration.max_batched_tokens == 2_048


def test_engine_configuration_explicit_values() -> None:
    """Create engine configuration with explicitly provided values."""
    configuration = EngineConfiguration(
        device="cuda",
        max_sequences=16,
        eos_token_id=2,
        block_size=32,
        num_blocks=512,
        max_batched_tokens=4_096,
    )

    assert configuration.device == "cuda"
    assert configuration.max_sequences == 16
    assert configuration.eos_token_id == 2
    assert configuration.block_size == 32
    assert configuration.num_blocks == 512
    assert configuration.max_batched_tokens == 4_096


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


@pytest.mark.parametrize(
    "block_size",
    [
        0,
        -1,
    ],
)
def test_non_positive_block_size_is_rejected(
    block_size: int,
) -> None:
    """Reject non-positive KV-cache block sizes."""
    with pytest.raises(ValueError, match="block_size must be greater than zero"):
        EngineConfiguration(block_size=block_size)


@pytest.mark.parametrize(
    "num_blocks",
    [
        0,
        -1,
    ],
)
def test_non_positive_num_blocks_is_rejected(
    num_blocks: int,
) -> None:
    """Reject non-positive KV-cache block counts."""
    with pytest.raises(ValueError, match="num_blocks must be greater than zero"):
        EngineConfiguration(num_blocks=num_blocks)


@pytest.mark.parametrize(
    "max_batched_tokens",
    [
        0,
        -1,
    ],
)
def test_non_positive_max_batched_tokens_is_rejected(
    max_batched_tokens: int,
) -> None:
    """Reject non-positive per-step token budgets."""
    with pytest.raises(
        ValueError,
        match="max_batched_tokens must be greater than zero",
    ):
        EngineConfiguration(max_batched_tokens=max_batched_tokens)


def test_max_batched_tokens_must_cover_max_sequences() -> None:
    """Reject a token budget smaller than the maximum decode batch."""
    with pytest.raises(
        ValueError,
        match="max_batched_tokens must be greater than or equal to max_sequences",
    ):
        EngineConfiguration(
            max_sequences=8,
            max_batched_tokens=7,
        )


def test_max_batched_tokens_equal_to_max_sequences_is_valid() -> None:
    """Accept a token budget exactly equal to the maximum decode batch."""
    configuration = EngineConfiguration(
        max_sequences=8,
        max_batched_tokens=8,
    )

    assert configuration.max_batched_tokens == configuration.max_sequences
