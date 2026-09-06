"""Paged KV-cache block allocation and mapping primitives."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from serving.cost_models.models import ModelSpecification


@dataclass
class Block:
    """Physical KV-cache block tracked by the allocator.

    Attributes:
        block_id: Stable identifier of the physical block within the pool.
        ref_count: Number of active references to the block. A value of zero
            means the block is free; a positive value means it is allocated.
    """

    block_id: int
    ref_count: int = 0


def kv_block_bytes(
    model: ModelSpecification,
    block_size: int,
    bytes_per_element: int,
) -> int:
    """Return KV-cache storage required by one physical block.

    Args:
        model: Model architecture used to determine the number of layers,
            KV heads, and head dimension.
        block_size: Number of token positions stored in one physical block.
        bytes_per_element: Number of bytes used by each KV-cache element.

    Returns:
        Storage required for one physical KV-cache block, in bytes.

    Raises:
        ValueError: If block_size or bytes_per_element is not positive.
    """
    if any(value <= 0 for value in (block_size, bytes_per_element)):
        raise ValueError(
            "block_size and bytes_per_element must be greater than zero.",
        )

    return block_size * model.kv_bytes_per_token(
        bytes_per_value=bytes_per_element,
    )


def blocks_needed(num_tokens: int, block_size: int) -> int:
    """Return the number of blocks required to store a token sequence.

    Args:
        num_tokens: Number of logical token positions to store.
        block_size: Number of token positions stored in one physical block.

    Returns:
        Number of physical blocks required to store the tokens.

    Raises:
        ValueError: If num_tokens is negative or block_size is not positive.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must not be negative.")

    if block_size <= 0:
        raise ValueError("block_size must be greater than zero.")

    if num_tokens == 0:
        return 0

    return (num_tokens + block_size - 1) // block_size


class BlockPool:
    """Manage a fixed collection of physical KV-cache blocks.

    The pool owns physical block identifiers, tracks their reference counts,
    and maintains a FIFO queue of blocks available for allocation.

    A block with a reference count of zero is free, while a positive reference
    count means the block is allocated. Multiple references may point to the
    same allocated block.
    """

    def __init__(self, num_blocks: int) -> None:
        """Initialize a pool of free physical KV-cache blocks.

        Args:
            num_blocks: Total number of physical blocks managed by the pool.

        Raises:
            ValueError: If num_blocks is not positive.
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")

        self.num_blocks = num_blocks

        self._blocks = [Block(block_id=block_id) for block_id in range(num_blocks)]

        self._free_block_ids = deque(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        """Return the number of blocks currently available for allocation."""
        return len(self._free_block_ids)

    @property
    def num_allocated_blocks(self) -> int:
        """Return the number of blocks currently allocated."""
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        """Return the fraction of physical blocks currently allocated."""
        return self.num_allocated_blocks / self.num_blocks

    def allocate(self, num_blocks: int) -> tuple[int, ...]:
        """Allocate physical blocks from the free-block queue.

        Allocation is atomic. If insufficient blocks are available, the pool
        remains unchanged.

        Args:
            num_blocks: Number of blocks to allocate.

        Returns:
            Allocated block identifiers in allocation order.

        Raises:
            RuntimeError: If there are not enough free blocks available.
            ValueError: If num_blocks is not positive.
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")

        if num_blocks > self.num_free_blocks:
            raise RuntimeError("Not enough free blocks available.")

        allocated_block_ids = tuple(
            self._free_block_ids.popleft() for _ in range(num_blocks)
        )

        for block_id in allocated_block_ids:
            self._blocks[block_id].ref_count = 1

        return allocated_block_ids

    def increment_ref_count(
        self,
        block_id: int,
    ) -> None:
        """Add one active reference to an allocated block.

        For the current allocator model, references may only be added to blocks
        that are already allocated. Cached-but-evictable blocks with zero
        references may require different semantics in later prefix-cache work.

        Args:
            block_id: Identifier of the block whose reference count is
                incremented.

        Raises:
            ValueError: If block_id is invalid or refers to a free block.
        """
        self._validate_block_id(block_id)

        block = self._blocks[block_id]

        if block.ref_count == 0:
            raise ValueError(
                "Cannot increment the reference count of a free block.",
            )

        block.ref_count += 1

    def free(
        self,
        block_id: int,
    ) -> None:
        """Release one reference to a physical block.

        The block returns to the free-block queue only when its final reference
        is released.

        Args:
            block_id: Identifier of the block to release.

        Raises:
            ValueError: If block_id is invalid or the block is already free.
        """
        self._validate_block_id(block_id)

        block = self._blocks[block_id]

        if block.ref_count == 0:
            raise ValueError("Cannot free a block that is already free.")

        block.ref_count -= 1

        if block.ref_count == 0:
            self._free_block_ids.append(block_id)

    def free_many(
        self,
        block_ids: Iterable[int],
    ) -> None:
        """Release one reference for each supplied physical block.

        Args:
            block_ids: Block identifiers whose references should be released.
        """
        for block_id in block_ids:
            self.free(block_id)

    def _validate_block_id(self, block_id: int) -> None:
        """Validate that a physical block identifier belongs to the pool.

        Args:
            block_id: Physical block identifier to validate.

        Raises:
            ValueError: If block_id does not belong to this pool.
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id: {block_id!r}.")


class BlockTable:
    """Map logical request blocks to physical KV-cache blocks.

    The table records the ordered physical block identifiers allocated to each
    request. Logical sequence length is intentionally not stored here; callers
    provide token counts or positions so sequence state remains the single
    source of truth.
    """

    def __init__(self, pool: BlockPool, block_size: int) -> None:
        """Initialize a block table backed by a physical block pool.

        Args:
            pool: Physical block pool used for allocation and release.
            block_size: Number of token positions stored in each physical block.

        Raises:
            ValueError: If block_size is not positive.
        """
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero.")

        self.pool = pool
        self.block_size = block_size
        self._block_tables: dict[str, list[int]] = {}

    def allocate_for_request(
        self,
        request_id: str,
        num_tokens: int,
    ) -> tuple[int, ...]:
        """Allocate physical blocks required by a request.

        Args:
            request_id: Unique identifier of the request.
            num_tokens: Number of logical tokens initially stored.

        Returns:
            Physical block identifiers allocated to the request.

        Raises:
            ValueError: If request_id already exists or num_tokens is not
                positive.
            RuntimeError: If the pool cannot satisfy the allocation.
        """
        if request_id in self._block_tables:
            raise ValueError(f"request_id {request_id!r} already has a block table.")

        if num_tokens <= 0:
            raise ValueError("num_tokens must be greater than zero.")

        num_blocks = blocks_needed(
            num_tokens=num_tokens,
            block_size=self.block_size,
        )

        block_ids = self.pool.allocate(num_blocks=num_blocks)

        self._block_tables[request_id] = list(block_ids)

        return block_ids

    def blocks_for_request(
        self,
        request_id: str,
    ) -> tuple[int, ...]:
        """Return the physical blocks assigned to a request.

        Args:
            request_id: Identifier of the request.

        Returns:
            Physical block identifiers in logical block order.

        Raises:
            ValueError: If request_id is unknown.
        """
        self._validate_request_id(request_id)

        return tuple(self._block_tables[request_id])

    def free_request(self, request_id: str) -> None:
        """Release all physical blocks assigned to a request.

        Args:
            request_id: Identifier of the request to release.

        Raises:
            ValueError: If request_id is unknown.
        """
        self._validate_request_id(request_id)

        block_ids = self._block_tables[request_id]

        self.pool.free_many(block_ids)

        del self._block_tables[request_id]

    def append_token(
        self,
        request_id: str,
        token_position: int,
    ) -> None:
        """Ensure storage exists for a logical token position.

        The logical block index is derived entirely from token_position rather
        than from an internal token counter. Calling the method repeatedly for
        an already mapped position is therefore idempotent.

        Args:
            request_id: Identifier of the request being extended.
            token_position: Zero-based logical position of the token.

        Raises:
            ValueError: If request_id is unknown, token_position is negative,
                or the position skips one or more logical blocks.
            RuntimeError: If a required physical block cannot be allocated.
        """
        self._validate_request_id(request_id)

        if token_position < 0:
            raise ValueError("token_position must not be negative.")

        block_ids = self._block_tables[request_id]

        logical_block_index = token_position // self.block_size

        if logical_block_index < len(block_ids):
            return

        if logical_block_index > len(block_ids):
            raise ValueError("token_position skips one or more logical blocks.")

        new_block_id = self.pool.allocate(1)[0]

        block_ids.append(new_block_id)

    def slot_for_position(
        self,
        request_id: str,
        token_position: int,
    ) -> int:
        """Return the flat physical slot for a logical token position.

        Args:
            request_id: Identifier of the request.
            token_position: Zero-based logical position of the token.

        Returns:
            Flat physical KV-cache slot corresponding to the token position.

        Raises:
            ValueError: If request_id is unknown, token_position is negative,
                or no physical block exists for the position.
        """
        self._validate_request_id(request_id)

        if token_position < 0:
            raise ValueError("token_position must not be negative.")

        block_ids = self._block_tables[request_id]

        logical_block_index = token_position // self.block_size

        if logical_block_index >= len(block_ids):
            raise ValueError(
                f"No physical block exists for token position: {token_position!r}",
            )

        block_offset = token_position % self.block_size
        physical_block_id = block_ids[logical_block_index]

        return physical_block_id * self.block_size + block_offset

    def fragmentation_slots(
        self,
        request_id: str,
        num_tokens: int,
    ) -> int:
        """Return unused token slots inside a request's allocated blocks.

        Args:
            request_id: Identifier of the request.
            num_tokens: Number of logical tokens currently stored.

        Returns:
            Number of allocated token slots that are currently unused.

        Raises:
            ValueError: If request_id is unknown or num_tokens is inconsistent
                with the request's allocated block capacity.
        """
        self._validate_request_id(request_id)

        if num_tokens <= 0:
            raise ValueError("num_tokens must be greater than zero.")

        block_ids = self._block_tables[request_id]

        required_blocks = blocks_needed(
            num_tokens=num_tokens,
            block_size=self.block_size,
        )

        if required_blocks != len(block_ids):
            raise ValueError(
                "num_tokens is inconsistent with the allocated blocks.",
            )

        allocated_slots = len(block_ids) * self.block_size

        return allocated_slots - num_tokens

    def _validate_request_id(self, request_id: str) -> None:
        """Validate that a request has an entry in the block table.

        Args:
            request_id: Request identifier to validate.

        Raises:
            ValueError: If request_id is unknown.
        """
        if request_id not in self._block_tables:
            raise ValueError(f"Unknown request_id: {request_id!r}.")
