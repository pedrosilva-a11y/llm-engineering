"""Sequence state models for the LLM inference engine."""

from dataclasses import dataclass, field
from enum import StrEnum

from serving.engine.request import Request


class SequenceStatus(StrEnum):
    """Execution states for a generation sequence."""

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


class FinishReason(StrEnum):
    """Reason why generation for a sequence has completed."""

    EOS = "eos"
    LENGTH = "length"


@dataclass
class SequenceState:
    """Mutable execution state for one generation request.

    Attributes:
        request: Immutable request associated with this sequence.
        generated_token_ids: Tokens generated after the original prompt.
        status: Current execution status of the sequence.
        finish_reason: Reason generation completed, if the sequence is finished.
    """

    request: Request
    generated_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: FinishReason | None = None

    @property
    def num_generated_tokens(self) -> int:
        """Return the number of tokens generated so far."""
        return len(self.generated_token_ids)

    @property
    def current_length(self) -> int:
        """Return the total sequence length including prompt and generated tokens."""
        return self.num_generated_tokens + len(self.request.prompt_token_ids)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """Return prompt and generated token identifiers as one sequence."""
        return self.request.prompt_token_ids + tuple(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        """Return whether generation has completed.

        The finish reason is the source of truth for terminal state.
        """
        return self.finish_reason is not None

    def append_token(self, token_id: int) -> None:
        """Append one generated token to the sequence.

        Args:
            token_id: Generated token identifier.

        Raises:
            ValueError: If the sequence is already finished.
        """
        if self.is_finished:
            raise ValueError("Cannot append a token to a finished sequence.")

        self.generated_token_ids.append(token_id)

    def mark_running(self) -> None:
        """Mark the sequence as actively running.

        Raises:
            RuntimeError: If the sequence is not defined as waiting or preempted.
        """
        if self.status not in {
            SequenceStatus.WAITING,
            SequenceStatus.PREEMPTED,
        }:
            raise RuntimeError(
                "Only a waiting or preempted sequence can be marked running.",
            )

        self.status = SequenceStatus.RUNNING

    def mark_preempted(self) -> None:
        """Mark a running sequence as preempted.

        Raises:
            RuntimeError: If the sequence is not currently running.
        """
        if self.status != SequenceStatus.RUNNING:
            raise RuntimeError("Only a running sequence can be preempted.")

        self.status = SequenceStatus.PREEMPTED

    def mark_finished(self, reason: FinishReason) -> None:
        """Mark generation as completed.

        Args:
            reason: Reason generation terminated.

        Raises:
            ValueError: If the sequence is already finished.
        """
        if self.is_finished:
            raise ValueError("Cannot finish an already finished sequence.")

        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason
