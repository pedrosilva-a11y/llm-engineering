"""Configurable streaming stub server for benchmark validation."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class StubServerConfiguration:
    """Configure deterministic timing behavior for the benchmark stub server.

    Attributes:
        first_token_delay_seconds: Delay before the first output token is emitted.
        inter_token_delay_seconds: Delay between consecutive output tokens.
        token_id: Token identifier emitted by the stub server.
    """

    first_token_delay_seconds: float = 0.1
    inter_token_delay_seconds: float = 0.03
    token_id: int = 1

    def __post_init__(self) -> None:
        """Validate stub server configuration."""
        if self.first_token_delay_seconds < 0:
            raise ValueError(
                "first_token_delay_seconds must be greater than or equal to zero."
            )

        if self.inter_token_delay_seconds < 0:
            raise ValueError(
                "inter_token_delay_seconds must be greater than or equal to zero."
            )

        if self.token_id < 0:
            raise ValueError("token_id must be greater than or equal to zero.")


class CompletionRequest(BaseModel):
    """Request accepted by the benchmark completion stub.

    Attributes:
        prompt_token_ids: Input token identifiers supplied to the server.
        max_tokens: Number of output tokens to emit.
        stream: Whether streaming output is requested.
    """

    prompt_token_ids: list[int] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    stream: bool = True


def create_stub_app(
    configuration: StubServerConfiguration | None = None,
) -> FastAPI:
    """Create a configurable FastAPI benchmark stub application.

    Args:
        configuration: Timing and output configuration for the stub server.

    Returns:
        FastAPI application exposing a streaming completions endpoint.
    """
    resolved_configuration = configuration or StubServerConfiguration()

    application = FastAPI()

    @application.post("/v1/completions")
    async def create_completion(request: CompletionRequest) -> StreamingResponse:
        """Stream deterministic output tokens for one completion request."""
        if not request.stream:
            raise HTTPException(
                status_code=400,
                detail="stub server supports streaming requests only.",
            )

        return StreamingResponse(
            _stream_completion(
                request=request,
                configuration=resolved_configuration,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return application


async def _stream_completion(
    *,
    request: CompletionRequest,
    configuration: StubServerConfiguration,
) -> AsyncIterator[str]:
    """Generate one deterministic SSE event per output token.

    Args:
        request: Completion request defining the number of output tokens.
        configuration: Timing and token configuration for the stub server.

    Yields:
        Server-sent events containing one output token followed by a DONE event.
    """
    await asyncio.sleep(configuration.first_token_delay_seconds)

    for token_index in range(request.max_tokens):
        if token_index > 0:
            await asyncio.sleep(configuration.inter_token_delay_seconds)

        finish_reason = "length" if token_index == request.max_tokens - 1 else None

        payload = {
            "choices": [
                {
                    "index": 0,
                    "token_id": configuration.token_id,
                    "finish_reason": finish_reason,
                },
            ],
        }

        yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"


app = create_stub_app()
