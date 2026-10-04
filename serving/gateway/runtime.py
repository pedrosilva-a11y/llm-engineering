"""Runtime composition for the inference HTTP gateway."""

from fastapi import FastAPI

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.gateway.app import create_gateway_app


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


def create_development_app() -> FastAPI:
    """Create the local development gateway application."""
    return create_gateway_app(create_development_engine())
