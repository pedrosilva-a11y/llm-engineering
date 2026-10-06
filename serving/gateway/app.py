"""OpenAI-style streaming HTTP gateway for the inference engine."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Literal, Self
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from serving.cost_models.catalog import HARDWARE_CATALOG, MODEL_CATALOG
from serving.engine.engine import Engine
from serving.engine.request import Request
from serving.engine.sequence import FinishReason, SequenceState
from serving.gateway.tokenizer import TextTokenizer


class HealthResponse(BaseModel):
    """Health status returned by the inference gateway.

    Attributes:
        status: Current gateway health state.
        service: Stable identifier for the gateway service.
    """

    status: Literal["ok"]
    service: str


class ModelCatalogEntry(BaseModel):
    """Model specification exposed through the gateway catalog.

    Attributes:
        name: Stable identifier for the model specification.
        n_layer: Number of transformer layers.
        d_model: Hidden-state dimensionality of the model.
        n_head: Number of attention heads.
        n_kv_head: Number of key-value attention heads.
        d_head: Dimensionality of each attention head.
        d_ff: Hidden dimensionality of the feed-forward network.
        vocab_size: Number of tokens in the model vocabulary.
        tied_embeddings: Whether input and output embedding weights are shared.
        norm_has_bias: Whether normalization layers include bias parameters.
    """

    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_head: int
    d_ff: int
    vocab_size: int
    tied_embeddings: bool
    norm_has_bias: bool


class HardwareCatalogEntry(BaseModel):
    """Hardware specification exposed through the gateway catalog.

    Attributes:
        name: Stable identifier for the hardware specification.
        peak_bf16_tflops: Peak BF16 compute throughput in tera floating-point
            operations per second.
        memory_bandwidth_tb_s: Peak memory bandwidth in terabytes per second.
        memory_capacity_gib: Available accelerator memory capacity in gibibytes.
    """

    name: str
    peak_bf16_tflops: float
    memory_bandwidth_tb_s: float
    memory_capacity_gib: float


class CatalogResponse(BaseModel):
    """Available model and hardware specifications.

    Attributes:
        models: Model specifications available in the catalog.
        hardware: Hardware specifications available in the catalog.
    """

    models: list[ModelCatalogEntry]
    hardware: list[HardwareCatalogEntry]


class CompletionRequest(BaseModel):
    """Request accepted by the completions endpoint.

    Attributes:
        model: Model identifier supplied by the client.
        prompt: Human-readable prompt to tokenize before inference.
        prompt_token_ids: Pre-tokenized prompt supplied directly to the engine.
        max_tokens: Maximum number of generated tokens.
        stream: Whether streaming output is requested.
        ignore_eos: Whether EOS tokens should be ignored as a termination condition
            until max_tokens is reached.
    """

    model: str = Field(min_length=1)
    prompt: str | None = None
    prompt_token_ids: list[int] | None = Field(default=None, min_length=1)
    max_tokens: int = Field(gt=0)
    stream: bool = True
    ignore_eos: bool = False

    @model_validator(mode="after")
    def validate_prompt_source(self) -> Self:
        """Require exactly one text or tokenized prompt representation."""
        if (self.prompt is None) == (self.prompt_token_ids is None):
            raise ValueError(
                "Exactly one of prompt or prompt_token_ids must be provided.",
            )

        return self


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


def create_gateway_app(engine: Engine, tokenizer: TextTokenizer) -> FastAPI:
    """Create an HTTP gateway around one shared inference engine.

    Args:
        engine: Inference engine serving completion requests.
        tokenizer: Text tokenizer used to encode prompts and decode generated tokens.

    Returns:
        FastAPI application exposing the inference gateway endpoints.
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

    @application.get("/v1/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        """Report whether the inference gateway is available.

        Returns:
            Health response describing the gateway availability and service identifier.
        """
        return HealthResponse(
            status="ok",
            service="llm-inference-gateway",
        )

    @application.get("/v1/catalog", response_model=CatalogResponse)
    async def get_catalog() -> CatalogResponse:
        """Return model and hardware specifications available to the application.

        Returns:
            Catalog response containing the available model and hardware specifications.
        """
        return CatalogResponse(
            models=[
                ModelCatalogEntry(
                    name=model.name,
                    n_layer=model.n_layer,
                    d_model=model.d_model,
                    n_head=model.n_head,
                    n_kv_head=model.n_kv_head,
                    d_head=model.d_head,
                    d_ff=model.d_ff,
                    vocab_size=model.vocab_size,
                    tied_embeddings=model.tied_embeddings,
                    norm_has_bias=model.norm_has_bias,
                )
                for model in MODEL_CATALOG.values()
            ],
            hardware=[
                HardwareCatalogEntry(
                    name=hardware.name,
                    peak_bf16_tflops=hardware.peak_bf16_tflops,
                    memory_bandwidth_tb_s=hardware.memory_bandwidth_tb_s,
                    memory_capacity_gib=hardware.memory_capacity_gib,
                )
                for hardware in HARDWARE_CATALOG.values()
            ],
        )

    @application.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest,
    ) -> StreamingResponse:
        """Submit a generation request and stream generated tokens.

        Args:
            request: Validated completion request containing either human-readable
                prompt text or pre-tokenized prompt identifiers.

        Returns:
            Streaming response containing generated completion events.

        Raises:
            HTTPException: If non-streaming generation is requested or prompt encoding,
                engine request validation, or scheduler submission rejects the input.
            RuntimeError: If tokenizer processing fails unexpectedly, a validated
                request violates the prompt-source invariant, or the engine stream
                broker has already been closed.
        """
        if not request.stream:
            raise HTTPException(
                status_code=400,
                detail="gateway supports streaming requests only.",
            )

        try:
            if request.prompt_token_ids is not None:
                prompt_token_ids = tuple(request.prompt_token_ids)
            elif request.prompt is not None:
                prompt_token_ids = tokenizer.encode_prompt(request.prompt)
            else:
                raise RuntimeError(
                    "validated completion request does not contain a prompt source.",
                )

            engine_request = Request(
                request_id=f"gateway-{uuid4().hex}",
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=request.max_tokens,
                ignore_eos=request.ignore_eos,
            )

            queue = broker.submit(engine_request)

        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return StreamingResponse(
            _stream_completion(
                request_id=engine_request.request_id,
                model_name=request.model,
                queue=queue,
                broker=broker,
                tokenizer=tokenizer,
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
    tokenizer: TextTokenizer,
) -> AsyncIterator[str]:
    """Serialize generated token events into a server-sent event stream.

    Generated token IDs are accumulated and decoded as a complete prefix so each
    emitted payload contains the cumulative human-readable completion text. The
    stream terminates with the standard ``[DONE]`` sentinel. If the consumer
    disconnects or closes the stream early, the associated engine request is
    cancelled through the broker.

    Args:
        request_id: Unique identifier of the generation request being streamed.
        model_name: Model identifier reported in each completion payload.
        queue: Request-specific queue containing generated token events. A ``None``
            item indicates that generation has completed and no further tokens
            will be emitted.
        broker: Shared engine stream broker used to cancel the request when the
            stream terminates or is closed by the consumer.
        tokenizer: Text tokenizer used to decode the accumulated generated token
            identifiers into cumulative completion text.

    Yields:
        Serialized server-sent event strings for each generated token, followed
        by a final ``data: [DONE]`` event when generation completes.
    """
    generated_token_ids: list[int] = []

    try:
        while True:
            event = await queue.get()

            if event is None:
                break

            generated_token_ids.append(event.token_id)
            cumulative_text = tokenizer.decode(generated_token_ids)

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
                        "cumulative_text": cumulative_text,
                        "token_id": event.token_id,
                        "finish_reason": finish_reason,
                    }
                ],
            }

            yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    finally:
        broker.cancel(request_id)
