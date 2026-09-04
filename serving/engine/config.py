"""Configuration models for the LLM inference engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfiguration:
    """Configuration for inference-engine execution.

    Attributes:
        device: Device used for model execution, such as ``cpu`` or ``cuda``.
        max_sequences: Maximum number of sequences that may run concurrently.
        eos_token_id: Token identifier representing the end of a sequence.
    """

    device: str = "cpu"
    max_sequences: int = 8
    eos_token_id: int = 0

    def __post_init__(self) -> None:
        """Validate engine configuration values.

        Raises:
            ValueError: If the device is empty, max_sequences is not positive,
                or eos_token_id is negative.
        """
        if not self.device.strip():
            raise ValueError("device must not be empty.")

        if self.max_sequences <= 0:
            raise ValueError("max_sequences must be greater than zero.")

        if self.eos_token_id < 0:
            raise ValueError("eos_token_id must not be negative.")
