"""Tests for the LLM inference HTTP gateway."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import httpx2
import pytest

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceStatus
from serving.gateway.app import (
    _EngineStreamBroker,
    _stream_completion,
    _TokenEvent,
    create_gateway_app,
)


@pytest.fixture
def anyio_backend() -> str:
    """Run async gateway tests with the asyncio backend."""
    return "asyncio"


@pytest.mark.anyio
async def test_completion_streams_generated_tokens_and_done() -> None:
    """Stream generated tokens followed by the terminal DONE event."""
    engine = _test_engine()

    async with _gateway_client(engine) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt_token_ids": [1, 2, 3],
                "max_tokens": 3,
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    events = _sse_data_events(response)

    assert events[-1] == "[DONE]"

    payloads = [json.loads(event) for event in events[:-1]]

    assert len(payloads) == 3

    assert [payload["choices"][0]["token_id"] for payload in payloads] == [5, 5, 5]

    assert [payload["choices"][0]["finish_reason"] for payload in payloads] == [
        None,
        None,
        "length",
    ]


@pytest.mark.anyio
async def test_completion_rejects_non_streaming_request() -> None:
    """Reject completion requests when streaming is disabled."""
    engine = _test_engine()

    async with _gateway_client(engine) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt_token_ids": [1],
                "max_tokens": 1,
                "stream": False,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == ("gateway supports streaming requests only.")


@pytest.mark.anyio
async def test_completion_rejects_invalid_http_payload() -> None:
    """Reject request payloads that violate the HTTP schema."""
    engine = _test_engine()

    async with _gateway_client(engine) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt_token_ids": [],
                "max_tokens": 1,
                "stream": True,
            },
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_completion_translates_engine_rejection_to_bad_request() -> None:
    """Translate engine request validation failures into HTTP 400 responses."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=2,
        max_sequences=1,
        max_batched_tokens=4,
    )
    engine = _test_engine(configuration=configuration)

    async with _gateway_client(engine) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "max_tokens": 1,
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert "exceeds max_batched_tokens" in response.json()["detail"]


@pytest.mark.anyio
async def test_broker_cancel_removes_stream_and_cancels_engine_request() -> None:
    """Remove stream state and cancel the corresponding engine request."""
    engine = _test_engine()
    broker = _EngineStreamBroker(engine)

    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=8,
    )

    try:
        queue = broker.submit(request)

        stream = broker._streams[request.request_id]
        sequence = stream.sequence

        assert stream.queue is queue
        assert engine.waiting_sequences == (sequence,)

        broker.cancel(request.request_id)

        assert request.request_id not in broker._streams
        assert not engine.waiting_sequences
        assert not engine.running_sequences
        assert engine.finished_sequences == (sequence,)

        assert sequence.status == SequenceStatus.FINISHED
        assert sequence.finish_reason == FinishReason.CANCELLED

        assert queue.get_nowait() is None

    finally:
        await broker.close()


@pytest.mark.anyio
async def test_stream_completion_cancels_request_when_consumer_closes() -> None:
    """Cancel unfinished engine work when the streaming consumer disconnects."""
    engine = _test_engine()
    broker = _EngineStreamBroker(engine)

    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=8,
    )

    try:
        queue = broker.submit(request)
        sequence = broker._streams[request.request_id].sequence

        # Enter the async generator without allowing the broker worker to finish
        # the request first.
        queue.put_nowait(
            _TokenEvent(
                token_id=5,
                finish_reason=None,
            )
        )

        stream = cast(
            AsyncGenerator[str, None],
            _stream_completion(
                request_id=request.request_id,
                model_name="stub-model",
                queue=queue,
                broker=broker,
            ),
        )

        first_chunk = await anext(stream)

        assert '"token_id": 5' in first_chunk
        assert sequence.finish_reason is None

        await stream.aclose()

        assert request.request_id not in broker._streams
        assert not engine.waiting_sequences
        assert not engine.running_sequences
        assert engine.finished_sequences == (sequence,)

        assert sequence.status == SequenceStatus.FINISHED
        assert sequence.finish_reason == FinishReason.CANCELLED

    finally:
        await broker.close()


@pytest.mark.anyio
async def test_broker_close_cancels_active_requests() -> None:
    """Cancel unfinished requests when the gateway broker shuts down."""
    engine = _test_engine()
    broker = _EngineStreamBroker(engine)

    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=8,
    )

    queue = broker.submit(request)
    sequence = broker._streams[request.request_id].sequence

    await broker.close()

    assert not broker._streams
    assert not engine.waiting_sequences
    assert not engine.running_sequences
    assert engine.finished_sequences == (sequence,)

    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.finish_reason == FinishReason.CANCELLED

    assert queue.get_nowait() is None


@asynccontextmanager
async def _gateway_client(
    engine: Engine,
) -> AsyncIterator[httpx2.AsyncClient]:
    """Create an in-process HTTP client for a gateway application."""
    application = create_gateway_app(engine)

    async with application.router.lifespan_context(application):
        transport = httpx2.ASGITransport(app=application)

        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client


def _test_engine(
    *,
    configuration: EngineConfiguration | None = None,
) -> Engine:
    """Create a deterministic CPU engine for gateway tests."""
    engine_configuration = configuration or EngineConfiguration()

    model_runner = DeterministicStubModelRunner(
        vocab_size=32,
        eos_token_id=engine_configuration.eos_token_id,
        generated_token_id=5,
        default_eos_after=1_000,
        device=engine_configuration.device,
    )

    return Engine(
        configuration=engine_configuration,
        model_runner=model_runner,
    )


def _sse_data_events(response: httpx2.Response) -> list[str]:
    """Extract SSE data payloads from an HTTP response."""
    return [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
