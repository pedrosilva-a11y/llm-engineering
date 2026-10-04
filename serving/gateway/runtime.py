"""Runtime composition for the inference HTTP gateway."""

from fastapi import FastAPI

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.gateway.app import create_gateway_app
from serving.gateway.tokenizer import QwenTextTokenizer, TextTokenizer


def create_development_engine() -> Engine:
    """Create the deterministic CPU engine used for local gateway development."""
    configuration = EngineConfiguration()

    model_runner = DeterministicStubModelRunner(
        vocab_size=32,
        eos_token_id=configuration.eos_token_id,
        generated_token_id=1,
        default_eos_after=1_000_000,
        device=configuration.device,
    )

    return Engine(
        configuration=configuration,
        model_runner=model_runner,
    )


def create_development_app(*, tokenizer: TextTokenizer | None = None) -> FastAPI:
    """Create the local development gateway application.

    The development application uses the deterministic CPU inference engine and
    accepts an optional tokenizer dependency. When no tokenizer is supplied, the
    pinned Qwen tokenizer is loaded so the local gateway exposes the same
    text-facing request and response behavior expected by the final demo runtime.

    Supplying a tokenizer is useful for tests, where a deterministic fake can be
    injected to avoid network access and Hugging Face dependency loading.

    Args:
        tokenizer: Optional text tokenizer used to encode incoming prompts and
            decode generated token sequences. If omitted, the pinned
            ``QwenTextTokenizer`` is loaded from pretrained artifacts.

    Returns:
        FastAPI application configured with the development inference engine and
        the selected tokenizer.
    """
    gateway_tokenizer = (
        tokenizer if tokenizer is not None else QwenTextTokenizer.from_pretrained()
    )

    return create_gateway_app(
        engine=create_development_engine(),
        tokenizer=gateway_tokenizer,
    )
