"""Hugging Face adapter for batched paged KV-cache execution."""

from typing import Any

import torch
from transformers.cache_utils import Cache

from serving.engine.batch_layout import BatchLayout
from serving.engine.paged_kv_cache import PagedKVCache


class HFBatchedPagedCache(Cache):
    """Adapt packed Hugging Face K/V updates to paged KV storage.

    Hugging Face receives all scheduled query tokens packed into one logical
    batch row:

        (1, kv_heads, total_query_tokens, head_dim)

    The engine's paged KV cache operates on one ``ModelExecution`` at a time
    using tensors shaped:

        (query_tokens, kv_heads, head_dim)

    This adapter splits the packed query region according to ``BatchLayout``,
    delegates phase-aware writes and gathers to ``PagedKVCache``, then
    concatenates each execution's complete logical history into one packed
    Hugging Face K/V tensor:

        (1, kv_heads, total_key_tokens, head_dim)

    ``PagedKVCache`` remains responsible for deciding which physical slots are
    written during PREFILL and DECODE.
    """

    def __init__(
        self,
        paged_cache: PagedKVCache,
        layout: BatchLayout,
    ) -> None:
        """Initialize the batched Hugging Face cache adapter.

        Args:
            paged_cache: Phase-aware paged KV-cache implementation.
            layout: Geometry and executions represented by the packed forward.
        """
        super().__init__(layers=[])

        self._paged_cache = paged_cache
        self._layout = layout

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write packed K/V states and return packed complete histories.

        Args:
            key_states: Newly computed Hugging Face key states with shape
                ``(1, kv_heads, total_query_tokens, head_dim)``.
            value_states: Newly computed value states with the same shape.
            layer_idx: Transformer layer being updated.
            *args: Additional Hugging Face cache arguments, unused.
            **kwargs: Additional Hugging Face cache arguments, unused.

        Returns:
            Full packed key and value histories with shape
            ``(1, kv_heads, total_key_tokens, head_dim)``.

        Raises:
            ValueError: If the Hugging Face states do not match the packed
                execution geometry.
            RuntimeError: If the paged cache returns history geometry that does
                not match ``BatchLayout``.
        """
        del args, kwargs

        self._validate_states(
            key_states=key_states,
            value_states=value_states,
        )

        gathered_keys: list[torch.Tensor] = []
        gathered_values: list[torch.Tensor] = []

        for index, execution in enumerate(self._layout.executions):
            query_offset = self._layout.query_offsets[index]
            query_length = self._layout.query_lengths[index]
            key_length = self._layout.key_lengths[index]

            query_end = query_offset + query_length

            execution_keys = key_states[
                0,
                :,
                query_offset:query_end,
                :,
            ].transpose(0, 1)

            execution_values = value_states[
                0,
                :,
                query_offset:query_end,
                :,
            ].transpose(0, 1)

            history_keys, history_values = self._paged_cache.update_and_gather(
                layer_index=layer_idx,
                execution=execution,
                keys=execution_keys,
                values=execution_values,
            )

            self._validate_gathered_history(
                keys=history_keys,
                values=history_values,
                expected_length=key_length,
                execution_index=index,
            )

            gathered_keys.append(history_keys)
            gathered_values.append(history_values)

        packed_keys = torch.cat(gathered_keys, dim=0)
        packed_values = torch.cat(gathered_values, dim=0)

        if packed_keys.shape[0] != self._layout.total_key_tokens:
            raise RuntimeError(
                "Packed key history length does not match BatchLayout: "
                f"expected {self._layout.total_key_tokens}, "
                f"got {packed_keys.shape[0]}."
            )

        if packed_values.shape[0] != self._layout.total_key_tokens:
            raise RuntimeError(
                "Packed value history length does not match BatchLayout: "
                f"expected {self._layout.total_key_tokens}, "
                f"got {packed_values.shape[0]}."
            )

        hf_keys = packed_keys.transpose(0, 1).unsqueeze(0)
        hf_values = packed_values.transpose(0, 1).unsqueeze(0)

        return hf_keys, hf_values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Reject scalar sequence-length queries for packed execution.

        Mixed packed execution contains independent logical sequence lengths,
        so there is no meaningful single cache length.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "HFBatchedPagedCache does not expose one scalar sequence length "
            f"for packed execution (layer {layer_idx})."
        )

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Reject scalar query-offset queries for packed execution.

        Packed requests use explicit sequence-local position identifiers.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "HFBatchedPagedCache does not expose one scalar query offset "
            f"for packed execution (layer {layer_idx})."
        )

    def get_mask_sizes(
        self,
        query_length: int,
        layer_idx: int,
    ) -> tuple[int, int]:
        """Reject cache-derived mask sizing for packed execution.

        The packed runner supplies the complete explicit four-dimensional
        attention mask.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "HFBatchedPagedCache requires an explicit packed attention mask; "
            "cache-derived mask sizing is unsupported "
            f"(query_length={query_length}, layer={layer_idx})."
        )

    def _validate_states(
        self,
        *,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Validate Hugging Face K/V tensors against packed query geometry."""
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
            raise ValueError("Packed execution must use one Hugging Face batch row.")

        query_length = key_states.shape[2]

        if query_length != self._layout.total_query_tokens:
            raise ValueError(
                "Packed KV query length does not match BatchLayout: "
                f"expected {self._layout.total_query_tokens}, "
                f"got {query_length}."
            )

    @staticmethod
    def _validate_gathered_history(
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        expected_length: int,
        execution_index: int,
    ) -> None:
        """Validate one execution's gathered paged KV history."""
        if keys.ndim != 3:
            raise RuntimeError(
                "Paged KV keys must have shape (tokens, kv_heads, head_dim)."
            )

        if values.ndim != 3:
            raise RuntimeError(
                "Paged KV values must have shape (tokens, kv_heads, head_dim)."
            )

        if keys.shape != values.shape:
            raise RuntimeError(
                "Paged KV key/value histories must have matching shapes."
            )

        if keys.shape[0] != expected_length:
            raise RuntimeError(
                "Paged KV history length does not match BatchLayout for "
                f"execution {execution_index}: expected {expected_length}, "
                f"got {keys.shape[0]}."
            )
