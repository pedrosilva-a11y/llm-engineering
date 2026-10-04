"""Tests for gateway runtime composition."""

import json

import httpx2
import pytest

from serving.engine.model_runner import DeterministicStubModelRunner
from serving.gateway.runtime import create_development_app, create_development_engine
from serving.gateway.tests.helpers import FakeTextTokenizer


@pytest.fixture
def anyio_backend() -> str:
    """Run async runtime tests with asyncio."""
    return "asyncio"


def test_create_development_engine_uses_cpu_stub_runner() -> None:
    """Build the local runtime with deterministic CPU runner."""
    engine = create_development_engine()

    assert engine.configuration.device == "cpu"
    assert isinstance(engine.model_runner, DeterministicStubModelRunner)


@pytest.mark.anyio
async def test_development_app_stream_completions() -> None:
    """Serve a completion through the fully composed local runtime."""
    tokenizer = FakeTextTokenizer()

    application = create_development_app(tokenizer=tokenizer)

    async with application.router.lifespan_context(application):
        transport = httpx2.ASGITransport(app=application)

        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "stub-model",
                    "prompt": "Hello",
                    "max_tokens": 2,
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert tokenizer.encoded_prompts == ["Hello"]

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert events[-1] == "[DONE]"

    payloads = [json.loads(event) for event in events[:-1]]

    assert [payload["choices"][0]["token_id"] for payload in payloads] == [1, 1]
    assert [payload["choices"][0]["cumulative_text"] for payload in payloads] == [
        "1",
        "11",
    ]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"
