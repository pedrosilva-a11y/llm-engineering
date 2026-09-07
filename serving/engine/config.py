"""Configuration models for the LLM inference engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfiguration:
    """Configuration for inference-engine execution.

    Attributes:
        device: Device used for model execution, such as ``cpu`` or ``cuda``.
        max_sequences: Maximum number of sequences that may run concurrently.
        eos_token_id: Token identifier representing the end of a sequence.
        block_size: Number of token positions stored in one physical KV-cache block.
        num_blocks: Number of physical KV-cache blocks available to the engine.
        max_batched_tokens: Maximum number of tokens scheduled in one engine step.
    """

    device: str = "cpu"
    max_sequences: int = 8
    eos_token_id: int = 0
    block_size: int = 16
    num_blocks: int = 256
    max_batched_tokens: int = 2_048

    def __post_init__(self) -> None:
        """Validate engine configuration values.

        Raises:
            ValueError: If device is empty, max_sequences, block_size, num_blocks, or
                max_batched_tokens are not positive, eos_token_id is negative, or
                max_batched_tokens is smaller than max_sequences.
        """
        if not self.device.strip():
            raise ValueError("device must not be empty.")

        if self.max_sequences <= 0:
            raise ValueError("max_sequences must be greater than zero.")

        if self.eos_token_id < 0:
            raise ValueError("eos_token_id must not be negative.")

        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero.")

        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")

        if self.max_batched_tokens <= 0:
            raise ValueError("max_batched_tokens must be greater than zero.")

        if self.max_batched_tokens < self.max_sequences:
            raise ValueError(
                "max_batched_tokens must be greater than or equal to max_sequences.",
            )
