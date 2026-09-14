"""Flat paged KV-cache tensor storage."""

import torch


class PagedKVStorage:
    """Store transformer key and value tensors in fixed physical slots.

    Physical slots are shared across all transformer layers. Each logical token
    position is mapped to one physical slot by the block table.

    Args:
        num_layers: Number of transformer layers.
        num_blocks: Number of physical KV-cache blocks.
        block_size: Number of token slots in each block.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimension of each key/value attention head.
        dtype: Tensor data type used for KV storage.
        device: Device on which KV-cache tensors are allocated.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialize fixed-capacity paged KV-cache storage."""
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")

        if block_size <= 0:
            raise ValueError("block_size must be greater than zero.")

        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be greater than zero.")

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than zero.")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.num_slots = num_blocks * block_size

        cache_shape = (
            num_layers,
            self.num_slots,
            num_kv_heads,
            head_dim,
        )

        self.key_cache = torch.empty(
            cache_shape,
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.empty(
            cache_shape,
            dtype=dtype,
            device=device,
        )

    def scatter(
        self,
        layer_index: int,
        slot_ids: tuple[int, ...],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write key and value tensors into physical slots for one layer.

        Args:
            layer_index: Transformer layer whose KV cache will be updated.
            slot_ids: Physical cache slots receiving the key/value tensors.
            keys: Key tensors with shape
                ``(num_tokens, num_kv_heads, head_dim)``.
            values: Value tensors with shape
                ``(num_tokens, num_kv_heads, head_dim)``.

        Raises:
            ValueError: If the layer index, slot identifiers, tensor shapes,
                dtypes, or devices are incompatible with the storage.
        """
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index must be within the configured layers.")

        if any(slot_id < 0 or slot_id >= self.num_slots for slot_id in slot_ids):
            raise ValueError("slot_ids must be within the configured physical slots.")

        expected_shape = (
            len(slot_ids),
            self.num_kv_heads,
            self.head_dim,
        )

        if tuple(keys.shape) != expected_shape:
            raise ValueError(f"keys must have shape {expected_shape}.")

        if tuple(values.shape) != expected_shape:
            raise ValueError(f"values must have shape {expected_shape}.")

        if keys.dtype != self.dtype:
            raise ValueError("keys dtype must match the KV-cache dtype.")

        if values.dtype != self.dtype:
            raise ValueError("values dtype must match the KV-cache dtype.")

        cache_device = self.key_cache.device

        if keys.device != cache_device:
            raise ValueError("keys device must match the KV-cache device.")

        if values.device != cache_device:
            raise ValueError("values device must match the KV-cache device.")

        slot_tensor = torch.tensor(
            slot_ids,
            dtype=torch.long,
            device=cache_device,
        )

        self.key_cache[layer_index].index_copy_(0, slot_tensor, keys)
        self.value_cache[layer_index].index_copy_(0, slot_tensor, values)

    def gather(
        self,
        layer_index: int,
        slot_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read key and value tensors from physical slots for one layer.

        Args:
            layer_index: Transformer layer whose KV cache will be read.
            slot_ids: Physical cache slots to gather, ordered by logical token
                position.

        Returns:
            Key and value tensors with shape
            ``(num_tokens, num_kv_heads, head_dim)`` in the same logical order
            as ``slot_ids``.

        Raises:
            ValueError: If the layer index or any slot identifier is outside the
                configured storage.
        """
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index must be within the configured layers.")

        if any(slot_id < 0 or slot_id >= self.num_slots for slot_id in slot_ids):
            raise ValueError("slot_ids must be within the configured physical slots.")

        slot_tensor = torch.tensor(
            slot_ids,
            dtype=torch.long,
            device=self.key_cache.device,
        )

        keys = self.key_cache[layer_index].index_select(0, slot_tensor)
        values = self.value_cache[layer_index].index_select(0, slot_tensor)

        return keys, values
