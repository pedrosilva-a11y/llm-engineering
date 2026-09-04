"""Run a small deterministic CPU generation demo."""

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.engine.request import Request


def main() -> None:
    """Run a deterministic multi-request generation example."""
    configuration = EngineConfiguration(
        device="cpu",
        max_sequences=2,
        eos_token_id=0,
    )

    model_runner = DeterministicStubModelRunner(
        vocab_size=10,
        eos_token_id=0,
        generated_token_id=5,
        eos_after_by_request={
            "request-a": 2,
            "request-b": 4,
        },
        device=configuration.device,
    )

    engine = Engine(
        configuration=configuration,
        model_runner=model_runner,
    )

    requests = [
        Request(
            request_id="request-a",
            prompt_token_ids=(1, 2),
            max_new_tokens=8,
        ),
        Request(
            request_id="request-b",
            prompt_token_ids=(3, 4),
            max_new_tokens=3,
        ),
    ]

    for request in requests:
        engine.submit(request)

    finished_sequences = engine.run_until_complete()

    for sequence in finished_sequences:
        print(
            f"{sequence.request.request_id}: "
            f"{sequence.generated_token_ids} -> "
            f"{sequence.finish_reason}"
        )


if __name__ == "__main__":
    main()
