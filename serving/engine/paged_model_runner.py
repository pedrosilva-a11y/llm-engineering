"""Paged KV-cache model runner for causal language models."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from transformers import PreTrainedModel

from serving.engine.attention_mask import (
    build_attention_mask,
    semantic_to_additive_attention_mask,
)
from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ModelExecution
from serving.engine.hf_batched_paged_cache import HFBatchedPagedCache
from serving.engine.packed_inputs import build_packed_inputs
from serving.engine.paged_kv_cache import PagedKVCache


class PagedModelRunner:
    """Run a causal LM with paged KV-cache storage.

    All scheduled executions are packed into one Hugging Face model forward.

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
        """Return next-token logits from one packed model forward.

        Args:
            executions: Scheduled model executions to evaluate.

        Returns:
            Next-token logits with shape ``(num_executions, vocab_size)``.

        Raises:
            ValueError: If no executions are provided.
        """
        if not executions:
            raise ValueError("At least one execution is required.")

        layout = BatchLayout(executions=tuple(executions))

        packed = build_packed_inputs(layout, device=self._device)

        semantic_mask = build_attention_mask(layout, device=self._device)

        model_attention_mask = semantic_to_additive_attention_mask(
            semantic_mask,
            dtype=self._model.dtype,
        )

        hf_cache = HFBatchedPagedCache(paged_cache=self._paged_cache, layout=layout)

        model_callable: Any = self._model

        with torch.inference_mode():
            outputs: Any = model_callable(
                input_ids=packed.input_ids.unsqueeze(0),
                position_ids=packed.position_ids.unsqueeze(0),
                attention_mask=model_attention_mask,
                past_key_values=hf_cache,
                use_cache=True,
            )

        logits = cast(torch.Tensor, outputs.logits)

        logit_indices = torch.tensor(
            layout.logit_indices,
            dtype=torch.long,
            device=self._device,
        )

        return logits[0].index_select(dim=0, index=logit_indices)
