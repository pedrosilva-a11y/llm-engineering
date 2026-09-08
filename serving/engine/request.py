"""Request models for the LLM inference engine."""

from dataclasses import dataclass, field

from serving.engine.sampling import SamplingParameters


@dataclass(frozen=True)
class Request:
    """Immutable input configuration for one generation request.

    Attributes:
        request_id: Unique identifier for the request.
        prompt_token_ids: Tokenized prompt provided to the model.
        max_new_tokens: Maximum number of tokens generated after the prompt,
            including EOS if EOS is generated.
        arrival_time: Request arrival timestamp used for ordering metrics.
        sampling_parameters: Token-selection configuration used during generation.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_time: float = 0.0
    sampling_parameters: SamplingParameters = field(
        default_factory=SamplingParameters,
    )

    def __post_init__(self) -> None:
        """Validate request values.

        Raises:
            ValueError: If the request identifier or prompt is empty, or if
                max_new_tokens is not positive.
        """
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty.")

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty.")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
