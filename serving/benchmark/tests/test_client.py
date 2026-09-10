"""Tests for the async LLM serving benchmark client."""

import asyncio
import time

import httpx2
import pytest

from serving.benchmark.client import (
    BenchmarkClient,
    BenchmarkClientConfiguration,
    _event_contains_output,
)
from serving.benchmark.stub_server import StubServerConfiguration, create_stub_app
from serving.benchmark.workload import WorkloadRequest


@pytest.fixture
def anyio_backend() -> str:
    """Run async benchmark tests with the asyncio backend."""
    return "asyncio"


@pytest.mark.parametrize(
    (
        "field_name",
        "endpoint_url",
        "model_name",
        "max_concurrency",
        "timeout_seconds",
    ),
    (
        ("endpoint_url", "", "stub-model", 1, 30.0),
        ("model_name", "http://testserver/v1/completions", "", 1, 30.0),
        ("max_concurrency", "http://testserver/v1/completions", "stub-model", 0, 30.0),
        ("timeout_seconds", "http://testserver/v1/completions", "stub-model", 1, 0.0),
    ),
)
def test_benchmark_client_configuration_rejects_invalid_values(
    field_name: str,
    endpoint_url: str,
    model_name: str,
    max_concurrency: int,
    timeout_seconds: float,
) -> None:
    """Reject one invalid client configuration field."""
    with pytest.raises(ValueError, match=field_name):
        BenchmarkClientConfiguration(
            endpoint_url=endpoint_url,
            model_name=model_name,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.anyio
async def test_run_fixed_workload_measures_streamed_request() -> None:
    """Measure token arrivals from the streaming benchmark stub."""
    application = create_stub_app(
        StubServerConfiguration(
            first_token_delay_seconds=0.0,
            inter_token_delay_seconds=0.0,
            token_id=7,
        )
    )

    client = BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url="http://testserver/v1/completions",
            model_name="stub-model",
        ),
        transport=httpx2.ASGITransport(app=application),
    )

    execution = await client.run_fixed_workload(
        (
            WorkloadRequest(
                request_id="request-000000",
                prompt_token_ids=(1, 2, 3),
                max_tokens=3,
                arrival_offset_seconds=0.0,
            ),
        )
    )

    assert len(execution.request_metrics) == 1
    assert execution.benchmark_duration_ns > 0

    metrics = execution.request_metrics[0]

    assert metrics.output_tokens == 3
    assert len(metrics.itl_ms) == 2
    assert metrics.tpot_ms is not None
    assert metrics.ttft_ms >= 0.0
    assert metrics.e2e_ms >= metrics.ttft_ms


@pytest.mark.anyio
async def test_run_fixed_workload_rejects_empty_workload() -> None:
    """Reject fixed workload execution without requests."""
    client = _benchmark_client()

    with pytest.raises(ValueError, match="requests must not be empty"):
        await client.run_fixed_workload(())


@pytest.mark.anyio
async def test_run_fixed_workload_rejects_scheduled_arrivals() -> None:
    """Reject non-zero arrival offsets in fixed-concurrency execution."""
    client = _benchmark_client()

    request = WorkloadRequest(
        request_id="request-000000",
        prompt_token_ids=(1,),
        max_tokens=1,
        arrival_offset_seconds=0.1,
    )

    with pytest.raises(ValueError, match="fixed workloads must contain only zero"):
        await client.run_fixed_workload((request,))


@pytest.mark.anyio
async def test_run_fixed_workload_respects_max_concurrency() -> None:
    """Never exceed the configured number of in-flight requests."""
    transport = _ConcurrencyTrackingTransport()

    client = BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url="http://testserver/v1/completions",
            model_name="stub-model",
            max_concurrency=2,
        ),
        transport=transport,
    )

    requests = tuple(
        WorkloadRequest(
            request_id=f"request-{index:06d}",
            prompt_token_ids=(1,),
            max_tokens=1,
            arrival_offset_seconds=0.0,
        )
        for index in range(6)
    )

    execution = await client.run_fixed_workload(requests)

    assert len(execution.request_metrics) == 6
    assert transport.max_active_requests == 2


@pytest.mark.anyio
async def test_run_scheduled_workload_rejects_empty_workload() -> None:
    """Reject scheduled workload execution without requests."""
    client = _benchmark_client()

    with pytest.raises(ValueError, match="requests must not be empty"):
        await client.run_scheduled_workload(())


@pytest.mark.anyio
async def test_run_scheduled_workload_executes_all_requests() -> None:
    """Execute all requests in a scheduled workload."""
    transport = _ConcurrencyTrackingTransport()

    client = BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url="http://testserver/v1/completions",
            model_name="stub-model",
            max_concurrency=2,
        ),
        transport=transport,
    )

    requests = (
        WorkloadRequest(
            request_id="request-000000",
            prompt_token_ids=(1,),
            max_tokens=1,
            arrival_offset_seconds=0.0,
        ),
        WorkloadRequest(
            request_id="request-000001",
            prompt_token_ids=(1,),
            max_tokens=1,
            arrival_offset_seconds=0.0,
        ),
    )

    execution = await client.run_scheduled_workload(requests)

    assert len(execution.request_metrics) == 2
    assert execution.benchmark_duration_ns > 0


@pytest.mark.anyio
async def test_run_scheduled_workloads_honors_arrival_offset() -> None:
    """Delay request execution until its configured arrival offset."""
    transport = _ArrivalTrackingTransport()

    client = BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url="http://testserver/v1/completions",
            model_name="stub-model",
        ),
        transport=transport,
    )

    requests = (
        WorkloadRequest(
            request_id="request=000000",
            prompt_token_ids=(1,),
            max_tokens=1,
            arrival_offset_seconds=0.0,
        ),
        WorkloadRequest(
            request_id="request-000001",
            prompt_token_ids=(1,),
            max_tokens=1,
            arrival_offset_seconds=0.05,
        ),
    )

    await client.run_scheduled_workload(requests)

    assert len(transport.request_start_ns) == 2

    arrival_gap_seconds = (
        transport.request_start_ns[1] - transport.request_start_ns[0]
    ) / 1_000_000_000

    assert arrival_gap_seconds >= 0.04


@pytest.mark.anyio
async def test_run_fixed_workload_rejects_missing_done_event() -> None:
    """Reject a stream that closes without the DONE event."""
    transport = _StaticStreamingTransport('data: {"choices": [{"token_id": 1}]}\n\n')
    client = _benchmark_client(transport=transport)

    with pytest.raises(RuntimeError, match="ended without a DONE event"):
        await client.run_fixed_workload((_workload_request(),))


@pytest.mark.anyio
async def test_run_fixed_workload_rejects_stream_without_output() -> None:
    """Reject a completed stream that contains no generated output."""
    transport = _StaticStreamingTransport("data: [DONE]\n\n")
    client = _benchmark_client(transport=transport)

    with pytest.raises(RuntimeError, match="completed without output tokens"):
        await client.run_fixed_workload((_workload_request(),))


@pytest.mark.anyio
async def test_run_fixed_workload_rejects_invalid_json_event() -> None:
    """Reject malformed JSON received in an SSE data event."""
    transport = _StaticStreamingTransport("data: not-json\n\ndata: [DONE]\n\n")
    client = _benchmark_client(transport=transport)

    with pytest.raises(RuntimeError, match="received an invalid JSON SSE payload"):
        await client.run_fixed_workload((_workload_request(),))


@pytest.mark.parametrize(
    ("data", "expected"),
    (
        ('{"choices": [{"token_id": 1}]}', True),
        ('{"choices": [{"text": "hello"}]}', True),
        ('{"choices": [{"text": ""}]}', False),
        ('{"choices": []}', False),
        ('{"choices": "invalid"}', False),
        ('{"choices": [1]}', False),
        ('{"other": "value"}', False),
        ("[]", False),
    ),
)
def test_event_contains_output(data: str, expected: bool) -> None:
    """Identify SSE payloads that contain generated output."""
    assert _event_contains_output(data) is expected


def _benchmark_client(
    transport: httpx2.AsyncBaseTransport | None = None,
) -> BenchmarkClient:
    """Create a benchmark client for tests."""
    return BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url="http://testserver/v1/completions",
            model_name="stub-model",
        ),
        transport=transport,
    )


def _workload_request() -> WorkloadRequest:
    """Create one immediately available benchmark request."""
    return WorkloadRequest(
        request_id="request-000000",
        prompt_token_ids=(1,),
        max_tokens=1,
        arrival_offset_seconds=0.0,
    )


class _StaticStreamingTransport(httpx2.AsyncBaseTransport):
    """Return a predetermined SSE response."""

    def __init__(self, content: str) -> None:
        """Initialize the transport with fixed response content."""
        self._content = content

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Return the configured streaming response."""
        return httpx2.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=self._content.encode(),
            request=request,
        )


class _ConcurrencyTrackingTransport(httpx2.AsyncBaseTransport):
    """Track simultaneous requests entering the HTTP transport."""

    def __init__(self) -> None:
        """Initialize concurrency counters."""
        self.active_requests = 0
        self.max_active_requests = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Return a successful SSE response while tracking concurrency."""
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)

        await asyncio.sleep(0.01)

        self.active_requests -= 1

        return httpx2.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=(b'data: {"choices": [{"token_id": 1}]}\n\ndata: [DONE]\n\n'),
            request=request,
        )


class _ArrivalTrackingTransport(httpx2.AsyncBaseTransport):
    """Record when requests reach the HTTP transport."""

    def __init__(self) -> None:
        """Initialize recorded request timestamps."""
        self.request_start_ns: list[int] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Record request arrival and return a successful SSE response."""
        self.request_start_ns.append(time.perf_counter_ns())

        return httpx2.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices": [{"token_id": 1}]}\n\ndata: [DONE]\n\n',
            request=request,
        )
