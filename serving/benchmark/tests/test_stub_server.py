"""Tests for the configurable benchmark streaming stub server."""

import json

import pytest
from fastapi.testclient import TestClient

from serving.benchmark.stub_server import (
    StubServerConfiguration,
    create_stub_app,
)


@pytest.mark.parametrize(
    (
        "field_name",
        "first_token_delay_seconds",
        "inter_token_delay_seconds",
        "token_id",
    ),
    (
        ("first_token_delay_seconds", -0.1, 0.03, 1),
        ("inter_token_delay_seconds", 0.1, -0.03, 1),
        ("token_id", 0.1, 0.03, -1),
    ),
)
def test_stub_server_configuration_rejects_invalid_values(
    field_name: str,
    first_token_delay_seconds: float,
    inter_token_delay_seconds: float,
    token_id: int,
) -> None:
    """Reject negative delay and token configuration values."""
    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be greater than or equal to zero\.",
    ):
        StubServerConfiguration(
            first_token_delay_seconds=first_token_delay_seconds,
            inter_token_delay_seconds=inter_token_delay_seconds,
            token_id=token_id,
        )


def test_stub_server_streams_one_sse_event_per_output_token() -> None:
    """Emit one SSE event for each requested output token."""
    client = _stub_client(token_id=7)

    response = client.post(
        "/v1/completions",
        json={"prompt_token_ids": [1, 2, 3], "max_tokens": 3, "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    data_lines = _sse_data_lines(response.text)

    assert len(data_lines) == 4
    assert data_lines[-1] == "[DONE]"

    token_events = tuple(json.loads(data) for data in data_lines[:-1])

    assert tuple(event["choices"][0]["token_id"] for event in token_events) == (7, 7, 7)


def test_stub_server_marks_only_final_token_finished() -> None:
    """Mark only the final generated token with the length finish reason."""
    client = _stub_client()

    response = client.post(
        "/v1/completions",
        json={"prompt_token_ids": [1], "max_tokens": 3, "stream": True},
    )

    assert response.status_code == 200

    data_lines = _sse_data_lines(response.text)
    token_events = tuple(json.loads(data) for data in data_lines[:-1])

    finish_reasons = tuple(
        event["choices"][0]["finish_reason"] for event in token_events
    )

    assert finish_reasons == (None, None, "length")
    assert data_lines[-1] == "[DONE]"


@pytest.mark.parametrize(
    "payload",
    (
        {
            "prompt_token_ids": [],
            "max_tokens": 1,
            "stream": True,
        },
        {
            "prompt_token_ids": [1],
            "max_tokens": 0,
            "stream": True,
        },
    ),
)
def test_stub_server_rejects_invalid_completion_request(
    payload: dict[str, object],
) -> None:
    """Reject invalid prompt and output-token request values."""
    client = _stub_client()

    response = client.post("/v1/completions", json=payload)

    assert response.status_code == 422


def test_stub_server_uses_streaming_by_default() -> None:
    """Stream completions when the stream field is omitted."""
    client = _stub_client()

    response = client.post(
        "/v1/completions",
        json={"prompt_token_ids": [1], "max_tokens": 1},
    )

    assert response.status_code == 200
    assert _sse_data_lines(response.text)[-1] == "[DONE]"


def test_stub_server_rejects_non_streaming_request() -> None:
    """Reject completion requests that explicitly disable streaming."""
    client = _stub_client()

    response = client.post(
        "/v1/completions",
        json={"prompt_token_ids": [1], "max_tokens": 1, "stream": False},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "stub server supports streaming requests only.",
    }


def _stub_client(token_id: int = 1) -> TestClient:
    """Create a zero-delay HTTP client for the benchmark stub."""
    application = create_stub_app(
        StubServerConfiguration(
            first_token_delay_seconds=0.0,
            inter_token_delay_seconds=0.0,
            token_id=token_id,
        )
    )

    return TestClient(application)


def _sse_data_lines(response_text: str) -> tuple[str, ...]:
    """Extract data payloads from a server-sent event stream."""
    return tuple(
        line.removeprefix("data: ")
        for line in response_text.splitlines()
        if line.startswith("data: ")
    )
