"""Paged KV-cache model runner for causal language models."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from transformers import PreTrainedModel

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.hf_paged_cache import HFPagedCache
from serving.engine.paged_kv_cache import PagedKVCache


class PagedModelRunner:
    """Run a causal LM with paged KV-cache storage.

    Each scheduled execution is evaluated independently with Hugging Face batch
    size one. Prefill executions process the complete logical sequence history,
    while decode executions process only the newest token and recover prior
    attention state from the paged KV cache.

    Args:
        model: Pre-loaded Hugging Face causal language model.
        device: Device used for model inference.
        paged_cache: Phase-aware paged KV cache shared across executions.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        paged_cache: PagedKVCache,
    ) -> None:
        """Initialize the paged model runner."""
        self._model = model
        self._device = device
        self._paged_cache = paged_cache

        cast(torch.nn.Module, self._model).eval()

    def forward(self, executions: Sequence[ModelExecution]) -> torch.Tensor:
        """Return next-token logits for each paged model execution.

        Args:
            executions: Scheduled model executions to evaluate.

        Returns:
            Next-token logits with shape ``(batch_size, vocab_size)``.

        Raises:
            ValueError: If no executions are provided or an execution phase is
                unsupported.
        """
        if not executions:
            raise ValueError("At least one execution is required.")

        batch_logits: list[torch.Tensor] = []
        model_callable: Any = self._model

        with torch.inference_mode():
            for execution in executions:
                input_token_ids = self._input_token_ids(execution)
                input_ids = torch.tensor(
                    input_token_ids,
                    dtype=torch.long,
                    device=self._device,
                ).unsqueeze(0)

                hf_cache = HFPagedCache(
                    paged_cache=self._paged_cache,
                    execution=execution,
                )

                outputs: Any = model_callable(
                    input_ids=input_ids,
                    past_key_values=hf_cache,
                    use_cache=True,
                )

                logits = cast(torch.Tensor, outputs.logits)
                batch_logits.append(logits[0, -1, :])

        return torch.stack(batch_logits)

    @staticmethod
    def _input_token_ids(execution: ModelExecution) -> tuple[int, ...]:
        """Return token identifiers that should be evaluated for an execution."""
        if execution.phase is ExecutionPhase.PREFILL:
            return execution.sequence.all_token_ids

        if execution.phase is ExecutionPhase.DECODE:
            return execution.sequence.all_token_ids[-1:]

        raise ValueError(f"Unsupported execution phase: {execution.phase}")
