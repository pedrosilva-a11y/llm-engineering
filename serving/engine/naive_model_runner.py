"""Naive reference model runner for causal language models."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from transformers import PreTrainedModel

from serving.engine.sequence import SequenceState


class NaiveModelRunner:
    """Run a causal LM by recomputing each full sequence history.

    This runner intentionally performs one model forward pass per sequence and
    disables KV caching. It serves as a correctness reference for later optimized
    model runners.
    """

    def __init__(self, model: PreTrainedModel, device: torch.device) -> None:
        """Initialize the naive model runner.

        Args:
            model: Pre-loaded Hugging Face causal language model.
            device: Device used for model inference.
        """
        self._model = model
        self._device = device
        cast(torch.nn.Module, self._model).eval()

    def forward(self, sequences: Sequence[SequenceState]) -> torch.Tensor:
        """Return next-token logits for each sequence.

        Args:
            sequences: Sequence states to evaluate.

        Returns:
            Next-token logits with shape ``(batch_size, vocab_size)``.

        Raises:
            ValueError: If no sequences are provided.
        """
        if not sequences:
            raise ValueError("At least one sequence is required.")

        batch_logits: list[torch.Tensor] = []

        with torch.inference_mode():
            for sequence in sequences:
                input_ids = torch.tensor(
                    sequence.all_token_ids,
                    dtype=torch.long,
                    device=self._device,
                ).unsqueeze(0)

                outputs: Any = self._model(input_ids=input_ids, use_cache=False)

                logits = cast(torch.Tensor, outputs.logits)
                batch_logits.append(logits[0, -1, :])

        return torch.stack(batch_logits)
