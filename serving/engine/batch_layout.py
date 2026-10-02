"""Pure batch-layout metadata for packed model execution."""

from dataclasses import dataclass

from serving.engine.execution import ExecutionPhase, ModelExecution


@dataclass(frozen=True)
class BatchLayout:
    """Describe packed query and key geometry for model executions.

    The layout is intentionally device-free and tensor-free. It derives the logical
    query and key regions that each execution will occupy in a future packed model
    forward.

    Prefill executions contribute their complete current sequence as queries, while
    decode executions contribute only their newest token. Both phases expose key/value
    regions covering the complete current logical sequence history, while causal
    masking determines which key positions each query may attend to.

    Attributes:
        executions: Ordered model executions included in the packed batch.
    """

    executions: tuple[ModelExecution, ...]

    def __post_init__(self) -> None:
        """Validate that the batch contains at least one execution.

        Raises:
            ValueError: If the batch contains no executions or an execution uses an
                unsupported phase.
        """
        if not self.executions:
            raise ValueError("BatchLayout requires at least one execution.")

        for execution in self.executions:
            self._query_length(execution)

    @property
    def query_lengths(self) -> tuple[int, ...]:
        """Return the number of query tokens contributed by each execution."""
        return tuple(self._query_length(execution) for execution in self.executions)

    @property
    def key_lengths(self) -> tuple[int, ...]:
        """Return the full attention-history length for each execution."""
        return tuple(execution.sequence.current_length for execution in self.executions)

    @property
    def query_offsets(self) -> tuple[int, ...]:
        """Return each execution's starting offset in packed query space."""
        return self._offsets(self.query_lengths)

    @property
    def key_offsets(self) -> tuple[int, ...]:
        """Return each execution's starting offset in packed key space."""
        return self._offsets(self.key_lengths)

    @property
    def logit_indices(self) -> tuple[int, ...]:
        """Return packed query indices used for next-token logits.

        Each request consumes the logits produced by its final query token.
        """
        return tuple(
            offset + length - 1
            for offset, length in zip(
                self.query_offsets, self.query_lengths, strict=True
            )
        )

    @property
    def total_query_tokens(self) -> int:
        """Return the total number of query tokens in the packed batch."""
        return sum(self.query_lengths)

    @property
    def total_key_tokens(self) -> int:
        """Return the total number of key positions in the packed batch."""
        return sum(self.key_lengths)

    @staticmethod
    def _query_length(execution: ModelExecution) -> int:
        """Return the number of query tokens contributed by one execution.

        Prefill executions contribute the complete current sequence history as query
        tokens because the model must compute representations for every token in the
        sequence. Decode executions contribute only the newest token because earlier
        key and value states are recovered from the KV cache.

        Args:
            execution: Model execution whose phase and current sequence length are used
                to determine the number of query tokens.

        Returns:
            Number of query tokens contributed by the execution. This is the sequence's
            current length for prefill and ``1`` for decode.

        Raises:
            ValueError: If the execution phase is not supported.
        """
        if execution.phase is ExecutionPhase.PREFILL:
            return execution.sequence.current_length

        if execution.phase is ExecutionPhase.DECODE:
            return 1

        raise ValueError(f"Unsupported execution phase: {execution.phase}")

    @staticmethod
    def _offsets(lengths: tuple[int, ...]) -> tuple[int, ...]:
        """Return starting offsets for consecutive packed regions.

        Each value in ``lengths`` represents the size of one logical region in a packed
        sequence. The returned tuple contains the starting index of each region after
        placing them consecutively with no gaps.

        For example, lengths ``(3, 1, 2)`` produce offsets ``(0, 3, 4)``:

        ``[0, 1, 2] [3] [4, 5]```

        Args:
            lengths: Sizes of consecutive packed regions, in execution order.

        Returns:
            Starting offset for each region. The returned tuple has the same length as
            ``lengths``.
        """
        offsets: list[int] = []
        current_offset = 0

        for length in lengths:
            offsets.append(current_offset)
            current_offset += length

        return tuple(offsets)
