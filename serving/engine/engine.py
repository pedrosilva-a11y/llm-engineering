"""Core generation loop for the LLM inference engine."""

from collections import deque

from serving.engine.config import EngineConfiguration
from serving.engine.model_runner import ModelRunner
from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceState


class Engine:
    """Coordinate request admission and batched autoregressive generation.

    The engine owns the lifecycle of submitted generation requests. It keeps
    waiting, running, and finished sequences, admits requests up to the
    configured concurrency limit, executes active sequences as one batch, and
    applies terminal conditions after each generated token.

    Model execution is delegated to a ``ModelRunner`` so the control loop
    remains independent of the concrete model implementation and execution
    device.
    """

    def __init__(
        self,
        configuration: EngineConfiguration,
        model_runner: ModelRunner,
    ) -> None:
        """Initialize the inference engine.

        Args:
            configuration: Runtime configuration for the inference engine.
            model_runner: Batched model runner used for model execution.
        """
        self.configuration = configuration
        self.model_runner = model_runner

        self._waiting: deque[SequenceState] = deque()
        self._running: list[SequenceState] = []
        self._finished: list[SequenceState] = []

    @property
    def waiting_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences waiting for execution capacity.

        Returns:
            Waiting sequences in queue order.
        """
        return tuple(self._waiting)

    @property
    def running_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences currently admitted for execution.

        Returns:
            Sequences currently participating in autoregressive decoding.
        """
        return tuple(self._running)

    @property
    def finished_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences whose generation has completed.

        Returns:
            Sequences whose generation has completed.
        """
        return tuple(self._finished)

    @property
    def has_unfinished_requests(self) -> bool:
        """Return whether waiting or running requests remain.

        Returns:
            True when at least one waiting or running sequence remains.
        """
        return bool(self._waiting or self._running)

    def _contains_request_id(self, request_id: str) -> bool:
        """Return whether a request identifier is already known to the engine.

        Args:
            request_id: Request identifier to search for.

        Returns:
            True when the identifier belongs to a waiting, running, or finished
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

    def _admit_waiting_sequences(self) -> None:
        """Admit waiting sequences while execution capacity is available."""
        while self._waiting and len(self._running) < self.configuration.max_sequences:
            sequence = self._waiting.popleft()
            sequence.mark_running()
            self._running.append(sequence)

    def submit(self, request: Request) -> None:
        """Submit one generation request to the engine.

        The request is converted into mutable sequence state and placed in the
        waiting queue. If execution capacity is available, waiting sequences are
        admitted immediately.

        Args:
            request: Immutable generation request to submit.

        Raises:
            ValueError: If the request identifier has already been submitted.
        """
        if self._contains_request_id(request.request_id):
            raise ValueError(
                f"request_id {request.request_id!r} has already been submitted."
            )

        self._waiting.append(SequenceState(request=request))
        self._admit_waiting_sequences()

    def _finish_sequence(
        self,
        sequence: SequenceState,
        reason: FinishReason,
    ) -> None:
        """Mark a running sequence as finished.

        Args:
            sequence: Running sequence whose generation has completed.
            reason: Reason generation terminated.
        """
        sequence.mark_finished(reason)

    def step(self) -> tuple[SequenceState, ...]:
        """Advance active sequences by one autoregressive decoding step.

        Waiting sequences are admitted while capacity is available. If no
        sequences are running, model execution is skipped.

        Otherwise, all running sequences are processed in one batched model
        execution. One token is selected for each sequence and checked against
        EOS and maximum-generation stopping conditions.

        Returns:
            Sequences that completed during this decoding step.
        """
        self._admit_waiting_sequences()

        if not self._running:
            return ()

        running_snapshot = tuple(self._running)

        logits = self.model_runner.forward(running_snapshot)

        finished_this_step: list[SequenceState] = []

        for index, sequence in enumerate(running_snapshot):
            token_id = int(logits[index].argmax().item())

            sequence.append_token(token_id)

            if token_id == self.configuration.eos_token_id:
                self._finish_sequence(
                    sequence=sequence,
                    reason=FinishReason.EOS,
                )
                finished_this_step.append(sequence)

            elif sequence.num_generated_tokens >= sequence.request.max_new_tokens:
                self._finish_sequence(
                    sequence=sequence,
                    reason=FinishReason.LENGTH,
                )
                finished_this_step.append(sequence)

        self._running = [
            sequence for sequence in self._running if not sequence.is_finished
        ]

        self._finished.extend(finished_this_step)

        self._admit_waiting_sequences()

        return tuple(finished_this_step)

    def run_until_complete(self) -> tuple[SequenceState, ...]:
        """Run decoding steps until all submitted requests have completed.

        Returns:
            All finished sequences in completion order.
        """
        while self.has_unfinished_requests:
            self.step()

        return self.finished_sequences
