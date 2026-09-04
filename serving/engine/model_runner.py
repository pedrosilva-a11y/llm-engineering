"""Model-runner interfaces for the LLM inference engine."""

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch

from serving.engine.sequence import SequenceState


class ModelRunner(Protocol):
    """Interface for batched model execution."""

    def forward(
        self,
        sequences: Sequence[SequenceState],
    ) -> torch.Tensor:
        """Return next-token logits for a batch of sequences.

        Args:
            sequences: Active generation sequences to process.

        Returns:
            Logits with shape ``(batch_size, vocab_size)``.
        """
        ...


class DeterministicStubModelRunner:
    """Produces deterministic logits for engine control-flow testing.

    The runner emits a configured token until a sequence reaches its
    request-specific generation limit, after which the EOS token receives
    the highest logit.

    Args:
        vocab_size: Number of tokens represented in the output logits.
        eos_token_id: Token identifier used to terminate generation.
        generated_token_id: Token emitted before EOS.
        eos_after_by_request: Number of non-EOS tokens to emit before EOS becomes
            the next predicted token.
        default_eos_after: EOS threshold used for requests without an explicit
            entry in ``eos_after_by_request``.
        device: Device on which logits are created.
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        generated_token_id: int,
        eos_after_by_request: Mapping[str, int] | None = None,
        default_eos_after: int = 2,
        device: str = "cpu",
    ) -> None:
        """Initialize the deterministic model runner."""
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        if not 0 <= eos_token_id < vocab_size:
            raise ValueError("eos_token_id must be within the vocabulary.")

        if not 0 <= generated_token_id < vocab_size:
            raise ValueError("generated_token_id must be within the vocabulary.")

        if generated_token_id == eos_token_id:
            raise ValueError(
                "generated_token_id must be different from eos_token_id.",
            )

        if default_eos_after <= 0:
            raise ValueError("default_eos_after must be greater than zero.")

        thresholds = dict(eos_after_by_request or {})

        if any(value <= 0 for value in thresholds.values()):
            raise ValueError(
                "EOS thresholds must be greater than zero.",
            )

        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.generated_token_id = generated_token_id
        self.eos_after_by_request = thresholds
        self.default_eos_after = default_eos_after
        self.device = torch.device(device)

    def forward(
        self,
        sequences: Sequence[SequenceState],
    ) -> torch.Tensor:
        """Produce deterministic next-token logits for a sequence batch.

        Args:
            sequences: Active generation sequences to process.

        Returns:
            Logits with shape ``(batch_size, vocab_size)``.

        Raises:
            ValueError: If the sequence batch is empty.
        """
        if not sequences:
            raise ValueError("sequences must not be empty.")

        logits = torch.full(
            (len(sequences), self.vocab_size),
            fill_value=float("-inf"),
            device=self.device,
        )

        for index, sequence in enumerate(sequences):
            eos_after = self.eos_after_by_request.get(
                sequence.request.request_id,
                self.default_eos_after,
            )

            next_token_id = (
                self.eos_token_id
                if sequence.num_generated_tokens >= eos_after
                else self.generated_token_id
            )

            logits[index, next_token_id] = 0.0

        return logits
