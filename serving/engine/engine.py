"""Core generation loop for the LLM inference engine."""

import torch

from serving.engine.config import EngineConfiguration
from serving.engine.model_runner import ModelRunner
from serving.engine.request import Request
from serving.engine.sampling import Sampler
from serving.engine.scheduler import Scheduler
from serving.engine.sequence import FinishReason, SequenceState, SequenceStatus


class Engine:
    """Coordinate scheduling, model execution, and autoregressive generation.

    The engine delegates request queues, admission policy, and KV-cache lifecycle
    management to ``Scheduler``. It executes the work selected for each step as one
    batched model call, applies generated tokens to sequence state, and reports
    completed sequences.

    Model execution is delegated to ``ModelRunner`` so the control loop remains
    independent of the concrete model implementation and execution device.
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

        generator = torch.Generator(
            device=configuration.device,
        ).manual_seed(configuration.seed)

        self.sampler = Sampler(generator=generator)

        self.scheduler = Scheduler(configuration=configuration)

    @property
    def waiting_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences waiting for execution capacity.

        Returns:
            Waiting sequences in queue order.
        """
        return self.scheduler.waiting_sequences

    @property
    def running_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences currently admitted for execution.

        Returns:
            Sequences currently admitted for execution.
        """
        return self.scheduler.running_sequences

    @property
    def finished_sequences(self) -> tuple[SequenceState, ...]:
        """Return sequences whose generation has completed.

        Returns:
            Sequences whose generation has completed.
        """
        return self.scheduler.finished_sequences

    @property
    def has_unfinished_requests(self) -> bool:
        """Return whether waiting or running requests remain.

        Returns:
            True when at least one waiting or running sequence remains.
        """
        return self.scheduler.has_unfinished_requests

    def submit(self, request: Request) -> None:
        """Submit one generation request to the scheduler.

        Args:
            request: Immutable generation request to submit.
        """
        self.scheduler.submit(request=request)

    def step(self) -> tuple[SequenceState, ...]:
        """Advance scheduled sequences by one inference-engine step.

        The scheduler selects existing decode work and newly admitted prefill work.
        If no work is selected, model execution is skipped. Otherwise, all selected
        sequences are processed in one batched model call.

        KV-cache storage is reserved before appending generated tokens that will be
        needed by a future decode step. Terminal EOS or length-limited tokens do not
        reserve additional KV capacity. Completed sequences are finalized through the
        scheduler.

        If reactive KV-cache preemption removes a sequence that was already included in
        the current model batch, its computed output is discarded because the sequence
        must be re-prefilled before generation can continue.

        Returns:
            Sequences that completed during this engine step.
        """
        scheduler_output = self.scheduler.schedule()

        batch = (
            *scheduler_output.decode_sequences,
            *scheduler_output.prefill_sequences,
        )

        if not batch:
            return ()

        logits = self.model_runner.forward(batch)

        finished_this_step: list[SequenceState] = []

        for index, sequence in enumerate(batch):
            if sequence.status == SequenceStatus.PREEMPTED:
                continue

            token_id = self.sampler.sample_token(
                logits=logits[index],
                parameters=sequence.request.sampling_parameters,
            )

            finishes_with_eos = token_id == self.configuration.eos_token_id
            finishes_with_length = (
                sequence.num_generated_tokens + 1 >= sequence.request.max_new_tokens
            )

            if not finishes_with_eos and not finishes_with_length:
                self.scheduler.reserve_generated_token(sequence)

            sequence.append_token(token_id)

            if finishes_with_eos:
                self.scheduler.finish_sequence(
                    sequence=sequence,
                    reason=FinishReason.EOS,
                )
                finished_this_step.append(sequence)

            elif finishes_with_length:
                self.scheduler.finish_sequence(
                    sequence=sequence,
                    reason=FinishReason.LENGTH,
                )
                finished_this_step.append(sequence)

        return tuple(finished_this_step)

    def run_until_complete(self) -> tuple[SequenceState, ...]:
        """Run decoding steps until all submitted requests have completed.

        Returns:
            All finished sequences in completion order.
        """
        while self.has_unfinished_requests:
            self.step()

        return self.finished_sequences
