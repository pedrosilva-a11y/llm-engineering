"""Phase-aware paged KV-cache operations."""

import torch

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.paged_kv_storage import PagedKVStorage


class PagedKVCache:
    """Coordinate phase-aware KV-cache updates and reads.

    The cache translates model execution metadata into physical KV-cache
    operations while delegating tensor storage to ``PagedKVStorage``.

    Args:
        storage: Physical paged KV tensor storage.
    """

    def __init__(self, storage: PagedKVStorage) -> None:
        """Initialize the paged KV cache."""
        self.storage = storage

    def update_and_gather(
        self,
        layer_index: int,
        execution: ModelExecution,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV storage and gather the full logical attention history.

        During prefill, keys and values represent the complete current sequence
        and are written to every physical slot in ``execution.slot_ids``.

        During decode, keys and values represent only the newest token and are
        written to the final physical slot. The complete KV history is then
        gathered in logical token order.

        Args:
            layer_index: Transformer layer whose KV cache is updated and read.
            execution: Scheduled execution containing phase and physical slots.
            keys: Newly computed key tensors for this execution.
            values: Newly computed value tensors for this execution.

        Returns:
            Key and value tensors for the complete current sequence history.

        Raises:
            ValueError: If the execution phase is unsupported.
        """
        if execution.phase is ExecutionPhase.PREFILL:
            write_slot_ids = execution.slot_ids

        elif execution.phase is ExecutionPhase.DECODE:
            write_slot_ids = execution.slot_ids[-1:]

        else:
            raise ValueError(f"Unsupported execution phase: {execution.phase}")

        self.storage.scatter(
            layer_index=layer_index,
            slot_ids=write_slot_ids,
            keys=keys,
            values=values,
        )

        return self.storage.gather(
            layer_index=layer_index,
            slot_ids=execution.slot_ids,
        )
