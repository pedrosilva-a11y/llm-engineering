"""Hugging Face adapter for paged KV-cache storage."""

from typing import Any

import torch
from transformers.cache_utils import Cache

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.paged_kv_cache import PagedKVCache


class HFPagedCache(Cache):
    """Adapt Hugging Face cache operations to paged KV storage.

    This adapter currently supports single-sequence causal self-attention.
    Hugging Face tensors use shape:

        (batch, kv_heads, tokens, head_dim)

    while the paged KV cache stores tensors as:

        (tokens, kv_heads, head_dim)
    """

    def __init__(
        self,
        paged_cache: PagedKVCache,
        execution: ModelExecution,
    ) -> None:
        """Initialize the Hugging Face paged-cache adapter.

        Args:
            paged_cache: Phase-aware paged KV-cache implementation.
            execution: Metadata describing the current model execution.
        """
        super().__init__(layers=[])

        self._paged_cache = paged_cache
        self._execution = execution

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the number of tokens already cached before this query."""
        del layer_idx

        if self._execution.phase is ExecutionPhase.PREFILL:
            return 0

        if self._execution.phase is ExecutionPhase.DECODE:
            return self._execution.sequence.current_length - 1

        raise ValueError(f"Unsupported execution phase: {self._execution.phase}")

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Return the logical position where the current query begins."""
        return self.get_seq_length(layer_idx=layer_idx)

    def get_mask_sizes(
        self,
        query_length: int,
        layer_idx: int,
    ) -> tuple[int, int]:
        """Return key/value length and offset for causal-mask construction."""
        del layer_idx

        expected_query_length = self._expected_query_length()

        if query_length != expected_query_length:
            raise ValueError(
                "query_length does not match the current execution phase: "
                f"expected {expected_query_length}, got {query_length}."
            )

        return self._execution.sequence.current_length, 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V states and return the full logical KV history."""
        del args, kwargs

        self._validate_states(
            key_states=key_states,
            value_states=value_states,
        )

        storage_keys = key_states[0].transpose(0, 1)
        storage_values = value_states[0].transpose(0, 1)

        gathered_keys, gathered_values = self._paged_cache.update_and_gather(
            layer_index=layer_idx,
            execution=self._execution,
            keys=storage_keys,
            values=storage_values,
        )

        hf_keys = gathered_keys.transpose(0, 1).unsqueeze(0)
        hf_values = gathered_values.transpose(0, 1).unsqueeze(0)

        return hf_keys, hf_values

    def _expected_query_length(self) -> int:
        """Return the number of tokens expected from Hugging Face."""
        if self._execution.phase is ExecutionPhase.PREFILL:
            return self._execution.sequence.current_length

        if self._execution.phase is ExecutionPhase.DECODE:
            return 1

        raise ValueError(f"Unsupported execution phase: {self._execution.phase}")

    def _validate_states(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Validate Hugging Face K/V tensors for the Day 8 contract."""
        if key_states.ndim != 4:
            raise ValueError(
                "key_states must have shape (batch, kv_heads, tokens, head_dim)."
            )

        if value_states.ndim != 4:
            raise ValueError(
                "value_states must have shape (batch, kv_heads, tokens, head_dim)."
            )

        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have matching shapes.")

        if key_states.shape[0] != 1:
            raise ValueError("HFPagedCache currently supports batch size 1 only.")

        query_length = key_states.shape[2]
        expected_query_length = self._expected_query_length()

        if query_length != expected_query_length:
            raise ValueError(
                "KV query length does not match the current execution phase: "
                f"expected {expected_query_length}, got {query_length}."
            )
