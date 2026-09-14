"""Model execution metadata for scheduled inference work."""

from dataclasses import dataclass
from enum import StrEnum

from serving.engine.sequence import SequenceState


class ExecutionPhase(StrEnum):
    """Model execution phase for a scheduled sequence."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ModelExecution:
    """Describe one scheduled model execution.

    Slot identifiers represent the complete current sequence history in logical
    token order, regardless of execution phase. During decode, the final slot is the
    write location for the newest token while all slots represent the KV history
    available to attention.

    Attributes:
        sequence: Sequence being executed.
        phase: Whether the execution is prefill or decode.
        slot_ids: Physical KV-cache slots for every logical token position in the
            current sequence.
    """

    sequence: SequenceState
    phase: ExecutionPhase
    slot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate execution metadata.

        Raises:
            ValueError: If the number of slot identifiers does not match the
                sequence's current length, or if any slot identifier is negative.
        """
        if len(self.slot_ids) != self.sequence.current_length:
            raise ValueError("slot_ids length must equal the sequence current length.")

        if any(slot_id < 0 for slot_id in self.slot_ids):
            raise ValueError("slot_ids must not contain negative values.")
