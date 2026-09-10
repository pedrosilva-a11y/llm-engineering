"""OpenAI-compatible streaming HTTP gateway for the inference engine."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceState


class CompletionRequest(BaseModel):
    """Request accepted by the completions endpoint.

    Attributes:
        model: Model identifier supplied by the client.
        prompt_token_ids: Tokenized prompt supplied to the inference engine.
        max_tokens: Maximum number of generated tokens.
        stream: Whether streaming output is requested.
    """

    model: str = Field(min_length=1)
    prompt_token_ids: list[int] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    stream: bool = True


@dataclass(frozen=True)
class _TokenEvent:
    """One generated token emitted by the engine.

    Attributes:
        token_id: Generated token identifier.
        finish_reason: Terminal reason when this is the final token.
    """

    token_id: int
    finish_reason: FinishReason | None


@dataclass
class _RequestStream:
    """Track streaming state for one submitted request."""

    sequence: SequenceState
    queue: asyncio.Queue[_TokenEvent | None]
    emitted_tokens: int = 0


class _EngineStreamBroker:
    """Drive one shared engine and dispatch generated tokens to HTTP streams."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the broker around a shared inference engine."""
        self.engine = engine
        self._streams: dict[str, _RequestStream] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()
        self._closed = False

    def submit(
        self,
        request: Request,
    ) -> asyncio.Queue[_TokenEvent | None]:
        """Submit a request and return its output-event queue.

        Args:
            request: Generation request submitted to the shared engine.

        Returns:
            Queue receiving generated token events.

        Raises:
            RuntimeError: If the broker has already been closed.
        """
        if self._closed:
            raise RuntimeError("engine stream broker is closed.")

        sequence = self.engine.submit(request)

        queue: asyncio.Queue[_TokenEvent | None] = asyncio.Queue()

        self._streams[request.request_id] = _RequestStream(
            sequence=sequence,
            queue=queue,
        )

        self._ensure_worker()
        self._wake_event.set()

        return queue

    def cancel(self, request_id: str) -> None:
        """Cancel an active request and remove its stream.

        Args:
            request_id: Identifier of the request whose stream should be cancelled.
        """
        stream = self._streams.pop(request_id, None)

        if stream is None:
            return

        self.engine.cancel(request_id)
        stream.queue.put_nowait(None)

    async def close(self) -> None:
        """Stop the background engine worker and cancel active requests."""
        self._closed = True
        self._wake_event.set()

        for request_id in tuple(self._streams):
            self.cancel(request_id)

        if self._worker_task is None:
            return

        self._worker_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._worker_task

    def _ensure_worker(self) -> None:
        """Start the engine worker when it is not already running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        """Continuously step the shared engine while work is available."""
        while not self._closed:
            if not self.engine.has_unfinished_requests:
                self._wake_event.clear()
                await self._wake_event.wait()
                continue

            self.engine.step()
            self._dispatch_generated_tokens()

            # Give request handlers and streaming responses an opportunity to run
            # between autoregressive engine steps.
            await asyncio.sleep(0)

    def _dispatch_generated_tokens(self) -> None:
        """Dispatch newly generated engine tokens to request-specific queues."""
        completed_request_ids: list[str] = []

        for request_id, stream in tuple(self._streams.items()):
            sequence = stream.sequence

            new_tokens = sequence.generated_token_ids[stream.emitted_tokens :]

            for index, token_id in enumerate(new_tokens):
                absolute_index = stream.emitted_tokens + index

                is_final_token = (
                    sequence.is_finished
                    and absolute_index == len(sequence.generated_token_ids) - 1
                )

                finish_reason = sequence.finish_reason if is_final_token else None

                stream.queue.put_nowait(
                    _TokenEvent(
                        token_id=token_id,
                        finish_reason=finish_reason,
                    )
                )

            stream.emitted_tokens += len(new_tokens)

            if sequence.is_finished and stream.emitted_tokens == len(
                sequence.generated_token_ids
            ):
                stream.queue.put_nowait(None)
                completed_request_ids.append(request_id)

        for request_id in completed_request_ids:
            del self._streams[request_id]


def create_gateway_app(engine: Engine) -> FastAPI:
    """Create an HTTP gateway around one shared inference engine.

    Args:
        engine: Inference engine serving completion requests.

    Returns:
        FastAPI application exposing the completions endpoint.
    """
    broker = _EngineStreamBroker(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close the engine broker when the application shuts down."""
        try:
            yield
        finally:
            await broker.close()

    application = FastAPI(lifespan=lifespan)

    @application.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest,
    ) -> StreamingResponse:
        """Submit a generation request and stream generated tokens."""
        if not request.stream:
            raise HTTPException(
                status_code=400,
                detail="gateway supports streaming requests only.",
            )

        engine_request = Request(
            request_id=f"gateway-{uuid4().hex}",
            prompt_token_ids=tuple(request.prompt_token_ids),
            max_new_tokens=request.max_tokens,
        )

        try:
            queue = broker.submit(engine_request)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        return StreamingResponse(
            _stream_completion(
                request_id=engine_request.request_id,
                model_name=request.model,
                queue=queue,
                broker=broker,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return application


async def _stream_completion(
    *,
    request_id: str,
    model_name: str,
    queue: asyncio.Queue[_TokenEvent | None],
    broker: _EngineStreamBroker,
) -> AsyncIterator[str]:
    """Serialize generated token events as server-sent events."""
    try:
        while True:
            event = await queue.get()

            if event is None:
                break

            finish_reason = (
                event.finish_reason.value if event.finish_reason is not None else None
            )

            payload = {
                "id": request_id,
                "object": "text_completion.chunk",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "token_id": event.token_id,
                        "finish_reason": finish_reason,
                    }
                ],
            }

            yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    finally:
        broker.cancel(request_id)


def _create_default_engine() -> Engine:
    """Create the CPU engine used by the development gateway."""
    configuration = EngineConfiguration()

    model_runner = DeterministicStubModelRunner(
        vocab_size=32,
        eos_token_id=configuration.eos_token_id,
        generated_token_id=1,
        # Keep EOS unreachable in normal development requests so max_tokens
        # controls completion length deterministically.
        default_eos_after=1_000_000,
        device=configuration.device,
    )

    return Engine(
        configuration=configuration,
        model_runner=model_runner,
    )


app = create_gateway_app(_create_default_engine())
