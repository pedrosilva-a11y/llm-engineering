"""Scheduling and KV-cache lifecycle management for the inference engine."""

from collections import deque
from dataclasses import dataclass

from serving.engine.block import BlockPool, BlockTable, blocks_needed
from serving.engine.config import EngineConfiguration
from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceState


@dataclass(frozen=True)
class SchedulerOutput:
    """Work selected for one inference-engine step.

    Attributes:
        decode_sequences: Existing running sequences selected for decode.
        prefill_sequences: Waiting sequences newly admitted for prefill.
        num_batched_tokens: Total token work scheduled for the step.
    """

    decode_sequences: tuple[SequenceState, ...]
    prefill_sequences: tuple[SequenceState, ...]
    num_batched_tokens: int


class Scheduler:
    """Own request queues and KV-cache allocation state.

    The scheduler tracks waiting, running and finished sequences and owns the
    physical KV-cache block pool and request block table. Scheduling policy,
    admission, and block lifecycle decisions are centralized here so the engine
    can remain focused on model execution and generation results.
    """

    def __init__(self, configuration: EngineConfiguration) -> None:
        """Initialize scheduler queues and KV-cache state.

        Args:
            configuration: Runtime inference-engine configuration.
        """
        self.configuration = configuration

        self.block_pool = BlockPool(num_blocks=configuration.num_blocks)
        self.block_table = BlockTable(
            pool=self.block_pool,
            block_size=configuration.block_size,
        )

        self._waiting: deque[SequenceState] = deque()
        self._running: list[SequenceState] = []
        self._finished: list[SequenceState] = []

        self._num_preemptions = 0

    @property
    def waiting_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences waiting to be scheduled."""
        return tuple(self._waiting)

    @property
    def running_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences currently admitted for execution."""
        return tuple(self._running)

    @property
    def finished_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences whose generation has completed."""
        return tuple(self._finished)

    @property
    def num_preemptions(self) -> int:
        """Return the total number of sequence preemptions."""
        return self._num_preemptions

    @property
    def has_unfinished_requests(self) -> bool:
        """Return whether waiting or running requests remain."""
        return bool(self._waiting or self._running)

    def submit(self, request: Request) -> None:
        """Add a generation request to the waiting queue.

        Requests that can never be scheduled are rejected immediately rather
        than being allowed to remain permanently in the waiting queue.

        Args:
            request: Immutable generation request to submit.

        Raises:
            ValueError: If the request identifier is already known, the prompt
                exceeds the per-step token budget, the maximum non-terminal sequence
                length cannot fit in one re-prefill budget, or the maximum non-terminal
                sequence length requires more KV-cache blocks than the scheduler owns.
        """
        if self._contains_request_id(request.request_id):
            raise ValueError(
                f"request_id {request.request_id!r} has already been submitted.",
            )

        prompt_length = len(request.prompt_token_ids)

        if prompt_length > self.configuration.max_batched_tokens:
            raise ValueError(
                "Prompt length exceeds max_batched_tokens and cannot be "
                "prefilled without chunked prefill."
            )

        max_kv_backed_tokens = prompt_length + request.max_new_tokens - 1

        if max_kv_backed_tokens > self.configuration.max_batched_tokens:
            raise ValueError(
                "Maximum non-terminal sequence length exceeds max_batched_tokens "
                "and cannot be re-prefilled without chunked prefill."
            )

        max_required_blocks = blocks_needed(
            num_tokens=max_kv_backed_tokens,
            block_size=self.configuration.block_size,
        )

        if max_required_blocks > self.configuration.num_blocks:
            raise ValueError(
                "Maximum non-terminal sequence length requires more KV-cache blocks "
                "than the scheduler owns."
            )

        self._waiting.append(SequenceState(request=request))

    def _blocks_required_for_sequence(
        self,
        sequence: SequenceState,
    ) -> int:
        """Return KV-cache blocks required for the sequence's current token history."""
        return blocks_needed(
            num_tokens=sequence.current_length,
            block_size=self.configuration.block_size,
        )

    def _can_admit(self, sequence: SequenceState) -> bool:
        """Return whether a waiting sequence can be admitted.

        Admission requires both an available sequence slot and enough free
        KV-cache blocks for the sequence's current token history.

        Args:
            sequence: Waiting sequence being considered for admission.

        Returns:
            True when both concurrency and KV-cache capacity allow admission.
        """
        if len(self._running) >= self.configuration.max_sequences:
            return False

        required_blocks = self._blocks_required_for_sequence(sequence)

        return required_blocks <= self.block_pool.num_free_blocks

    def _admit_sequence(self, sequence: SequenceState) -> None:
        """Allocate current sequence storage and move a waiting sequence to running.

        Args:
            sequence: Waiting sequence to admit.

        Raises:
            RuntimeError: If the sequence is not at the head of the waiting queue
                or KV-cache allocation fails.
        """
        if not self._waiting or self._waiting[0] is not sequence:
            raise RuntimeError("Admission violated waiting-queue ordering.")

        self.block_table.allocate_for_request(
            request_id=sequence.request.request_id,
            num_tokens=sequence.current_length,
        )

        self._waiting.popleft()

        sequence.mark_running()
        self._running.append(sequence)

    def _admit_waiting_sequences(self) -> tuple[SequenceState, ...]:
        """Admit waiting sequences while execution capacity permits.

        This internal admission primitive respects concurrency and KV-cache
        capacity but does not enforce the aggregate per-step token budget.
        Production step scheduling should use ``schedule()``.

        Admission follows first-come-first-served order. If the oldest waiting
        request cannot fit in available KV-cache capacity, admission stops rather
        than skipping it.

        Returns:
            Sequences admitted during this call, in admission order.
        """
        admitted: list[SequenceState] = []

        while self._waiting:
            sequence = self._waiting[0]

            if not self._can_admit(sequence):
                break

            self._admit_sequence(sequence)
            admitted.append(sequence)

        return tuple(admitted)

    def schedule(self) -> SchedulerOutput:
        """Select work for one inference-engine step within the token budget.

            Existing running sequences are prioritized for decode, where each sequence
            consumes one token of scheduling budget. Remaining budget is used to admit
            waiting requests for prefill in first-come-first-served order.

            This method mutates scheduler state when prefill requests are admitted by
            allocating KV-cache blocks for their current token history and moving them
            from waiting to running.

            Prefill admission also respects sequence concurrency and available KV-cache
            capacity. If the oldest waiting request cannot fit the remaining token
            budget or available execution capacity, admission stops rather than skipping
            it.

        Returns:
            Decode and prefill sequences selected for the step together with the
            total number of scheduled tokens.
        """
        remaining_budget = self.configuration.max_batched_tokens

        decode_sequences = tuple(self._running)
        remaining_budget -= len(decode_sequences)

        prefill_sequences: list[SequenceState] = []

        while self._waiting and remaining_budget > 0:
            sequence = self._waiting[0]
            prefill_tokens = sequence.current_length

            if prefill_tokens > remaining_budget:
                break

            if not self._can_admit(sequence):
                break

            self._admit_sequence(sequence)
            prefill_sequences.append(sequence)
            remaining_budget -= prefill_tokens

        num_batched_tokens = len(decode_sequences) + sum(
            sequence.current_length for sequence in prefill_sequences
        )

        return SchedulerOutput(
            decode_sequences=decode_sequences,
            prefill_sequences=tuple(prefill_sequences),
            num_batched_tokens=num_batched_tokens,
        )

    def reserve_generated_token(self, sequence: SequenceState) -> None:
        """Reserve KV-cache storage for the next generated logical token.

        The token position is derived from the sequence's current length before the
        generated token is appended. If KV capacity is exhausted, an eligible
        running sequence is preempted and the reservation is retried.

        Args:
            sequence: Running sequence whose generated token requires KV storage.

        Raises:
            RuntimeError: If the sequence is finished, is not currently running,
                KV allocation fails for a reason other than capacity exhaustion,
                or no eligible preemption victim exists.
        """
        if sequence.is_finished:
            raise RuntimeError(
                "Cannot reserve KV-cache storage for a finished sequence.",
            )

        if not any(running is sequence for running in self._running):
            raise RuntimeError(
                "Cannot reserve KV-cache storage for a sequence that is not running.",
            )

        try:
            self.block_table.append_token(
                request_id=sequence.request.request_id,
                token_position=sequence.current_length,
            )
            return

        except RuntimeError as error:
            if "Not enough free blocks available" not in str(error):
                raise

        victim = self._select_preemption_victim(requester=sequence)

        if victim is None:
            raise RuntimeError(
                "Not enough KV-cache capacity and no eligible preemption victim exists."
            )

        self._preempt_sequence(victim)

        self.block_table.append_token(
            request_id=sequence.request.request_id,
            token_position=sequence.current_length,
        )

    def _release_sequence_resources(self, sequence: SequenceState) -> None:
        """Release KV-cache blocks and remove a sequence from running state.

        Args:
            sequence: Running sequence whose scheduler resources should be released.
        """
        self.block_table.free_request(
            request_id=sequence.request.request_id,
        )

        self._running = [
            running for running in self._running if running is not sequence
        ]

    def _preempt_sequence(self, sequence: SequenceState) -> None:
        """Preempt a running sequence and return it to the waiting queue.

        KV-cache blocks owned by the sequence are released while its generated
        token history is preserved for future recomputation.

        Args:
            sequence: Running sequence to preempt.

        Raises:
            RuntimeError: If the sequence is finished or is not currently running.
        """
        if sequence.is_finished:
            raise RuntimeError("Cannot preempt a finished sequence.")

        if not any(running is sequence for running in self._running):
            raise RuntimeError("Cannot preempt a sequence that is not running.")

        self._release_sequence_resources(sequence)

        sequence.mark_preempted()
        self._waiting.appendleft(sequence)
        self._num_preemptions += 1

    def _select_preemption_victim(
        self,
        requester: SequenceState,
    ) -> SequenceState | None:
        """Select the most recently scheduled eligible preemption victim.

        Args:
            requester: Sequence requesting additional KV-cache capacity.

        Returns:
            Last running sequence other than the requester, or None when no
            eligible victim exists.
        """
        for sequence in reversed(self._running):
            if sequence is not requester:
                return sequence

        return None

    def finish_sequence(
        self,
        sequence: SequenceState,
        reason: FinishReason,
    ) -> None:
        """Finish a running sequence and release its KV-cache blocks.

        Args:
            sequence: Running sequence whose generation has completed.
            reason: Reason generation terminated.

        Raises:
            RuntimeError: If the sequence is not currently running or is already
                finished.
        """
        if sequence.is_finished:
            raise RuntimeError("Cannot finish a sequence that is already finished.")

        if not any(running is sequence for running in self._running):
            raise RuntimeError("Cannot finish a sequence that is not running.")

        self._release_sequence_resources(sequence)

        sequence.mark_finished(reason)
        self._finished.append(sequence)

    def _contains_request_id(self, request_id: str) -> bool:
        """Return whether a request identifier is known to the scheduler.

        Args:
            request_id: Request identifier to search for.

        Returns:
            True if the identifier belongs to a waiting, running, or finished
            sequence.
        """
        return any(
            sequence.request.request_id == request_id
            for sequence in (
                *self._waiting,
                *self._running,
                *self._finished,
            )
        )
