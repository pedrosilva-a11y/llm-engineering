"""Naive reference model runner for causal language models."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from transformers import PreTrainedModel

from serving.engine.execution import ModelExecution


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

    def forward(self, executions: Sequence[ModelExecution]) -> torch.Tensor:
        """Return next-token logits for each model execution.

        Args:
            executions: Scheduled model executions to evaluate.

        Returns:
            Next-token logits with shape ``(batch_size, vocab_size)``.

        Raises:
            ValueError: If no executions are provided.
        """
        if not executions:
            raise ValueError("At least one execution is required.")

        batch_logits: list[torch.Tensor] = []

        with torch.inference_mode():
            for execution in executions:
                sequence = execution.sequence

                input_ids = torch.tensor(
                    sequence.all_token_ids,
                    dtype=torch.long,
                    device=self._device,
                ).unsqueeze(0)

                outputs: Any = self._model(input_ids=input_ids, use_cache=False)

                logits = cast(torch.Tensor, outputs.logits)
                batch_logits.append(logits[0, -1, :])

        return torch.stack(batch_logits)
