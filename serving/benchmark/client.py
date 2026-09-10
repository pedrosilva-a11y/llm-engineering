"""Async streaming client for LLM serving benchmarks."""

import asyncio
import json
import time
from dataclasses import dataclass

import httpx2

from serving.benchmark.metrics import RequestMetrics, compute_request_metrics
from serving.benchmark.workload import WorkloadRequest

_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class BenchmarkClientConfiguration:
    """Configure the HTTP benchmark client.

    Attributes:
        endpoint_url: Streaming completions endpoint to benchmark.
        model_name: Model identifier sent with completion requests.
        max_concurrency: Maximum number of requests allowed in flight.
        timeout_seconds: HTTP request timeout in seconds.
    """

    endpoint_url: str
    model_name: str
    max_concurrency: int = 1
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate benchmark client configuration."""
        if not self.endpoint_url:
            raise ValueError("endpoint_url must not be empty.")

        if not self.model_name:
            raise ValueError("model_name must not be empty.")

        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")


@dataclass(frozen=True)
class BenchmarkExecution:
    """Measurements collected from one benchmark execution.

    Attributes:
        request_metrics: Metrics collected for each completed request.
        benchmark_duration_ns: Wall-clock duration of workload execution.
    """

    request_metrics: tuple[RequestMetrics, ...]
    benchmark_duration_ns: int


class BenchmarkClient:
    """Execute streaming benchmark requests with bounded concurrency."""

    def __init__(
        self,
        configuration: BenchmarkClientConfiguration,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the benchmark client.

        Args:
            configuration: HTTP endpoint and concurrency configuration.
            transport: Optional HTTPX2 transport, primarily useful for tests.
        """
        self.configuration = configuration
        self._transport = transport

    async def run_fixed_workload(
        self,
        requests: tuple[WorkloadRequest, ...],
    ) -> BenchmarkExecution:
        """Execute an immediately available workload with fixed concurrency.

        All requests must have an arrival offset of zero. Arrival scheduling is
        handled separately by the Poisson execution path.

        Args:
            requests: Benchmark requests to execute.

        Returns:
            Per-request measurements and complete benchmark wall-clock duration.

        Raises:
            ValueError: If the workload is empty or contains scheduled arrivals.
        """
        if not requests:
            raise ValueError("requests must not be empty.")

        if any(request.arrival_offset_seconds != 0.0 for request in requests):
            raise ValueError(
                "fixed workloads must contain only zero arrival offsets.",
            )

        semaphore = asyncio.Semaphore(self.configuration.max_concurrency)
        timeout = httpx2.Timeout(self.configuration.timeout_seconds)

        async with httpx2.AsyncClient(
            timeout=timeout,
            transport=self._transport,
        ) as client:
            benchmark_start_ns = time.perf_counter_ns()

            request_metrics = await asyncio.gather(
                *(
                    self._execute_with_limit(
                        client=client,
                        semaphore=semaphore,
                        request=request,
                    )
                    for request in requests
                )
            )

            benchmark_end_ns = time.perf_counter_ns()

        return BenchmarkExecution(
            request_metrics=tuple(request_metrics),
            benchmark_duration_ns=benchmark_end_ns - benchmark_start_ns,
        )

    async def run_scheduled_workload(
        self,
        requests: tuple[WorkloadRequest, ...],
    ) -> BenchmarkExecution:
        """Execute a workload according to its configured arrival offsets.

        Args:
            requests: Benchmark requests with arrival offsets relative to the
                beginning of the benchmark.

        Returns:
            Per-request measurements and complete benchmark wall-clock duration.

        Raises:
            ValueError: If the workload is empty.
        """
        if not requests:
            raise ValueError("requests must not be empty.")

        semaphore = asyncio.Semaphore(self.configuration.max_concurrency)
        timeout = httpx2.Timeout(self.configuration.timeout_seconds)

        async with httpx2.AsyncClient(
            timeout=timeout,
            transport=self._transport,
        ) as client:
            benchmark_start_ns = time.perf_counter_ns()

            request_metrics = await asyncio.gather(
                *(
                    self._execute_at_offset(
                        client=client,
                        semaphore=semaphore,
                        request=request,
                        benchmark_start_ns=benchmark_start_ns,
                    )
                    for request in requests
                )
            )

            benchmark_end_ns = time.perf_counter_ns()

        return BenchmarkExecution(
            request_metrics=tuple(request_metrics),
            benchmark_duration_ns=benchmark_end_ns - benchmark_start_ns,
        )

    async def _execute_with_limit(
        self,
        *,
        client: httpx2.AsyncClient,
        semaphore: asyncio.Semaphore,
        request: WorkloadRequest,
    ) -> RequestMetrics:
        """Execute one request after acquiring a concurrency slot."""
        async with semaphore:
            return await self._execute_request(client=client, request=request)

    async def _execute_at_offset(
        self,
        *,
        client: httpx2.AsyncClient,
        semaphore: asyncio.Semaphore,
        request: WorkloadRequest,
        benchmark_start_ns: int,
    ) -> RequestMetrics:
        """Wait until a scheduled arrival time and execute one request."""
        scheduled_arrival_ns = benchmark_start_ns + int(
            request.arrival_offset_seconds * _NANOSECONDS_PER_SECOND
        )

        remaining_ns = scheduled_arrival_ns - time.perf_counter_ns()

        if remaining_ns > 0:
            await asyncio.sleep(remaining_ns / _NANOSECONDS_PER_SECOND)

        return await self._execute_with_limit(
            client=client,
            semaphore=semaphore,
            request=request,
        )

    async def _execute_request(
        self,
        *,
        client: httpx2.AsyncClient,
        request: WorkloadRequest,
    ) -> RequestMetrics:
        """Execute one streaming completion and measure token arrivals."""
        request_start_ns = time.perf_counter_ns()

        token_arrival_ns: list[int] = []
        request_end_ns: int | None = None
        done_received = False

        async with client.stream(
            "POST",
            self.configuration.endpoint_url,
            json={
                "model": self.configuration.model_name,
                "prompt_token_ids": list(request.prompt_token_ids),
                "max_tokens": request.max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                event_arrival_ns = time.perf_counter_ns()
                data = line.removeprefix("data: ")

                if data == "[DONE]":
                    if not done_received:
                        request_end_ns = event_arrival_ns
                        done_received = True

                    continue

                if done_received:
                    continue

                if _event_contains_output(data):
                    token_arrival_ns.append(event_arrival_ns)

        if request_end_ns is None:
            raise RuntimeError(
                f"request {request.request_id} ended without a DONE event.",
            )

        if not token_arrival_ns:
            raise RuntimeError(
                f"request {request.request_id} completed without output tokens.",
            )

        return compute_request_metrics(
            request_start_ns=request_start_ns,
            token_arrival_ns=tuple(token_arrival_ns),
            request_end_ns=request_end_ns,
        )


def _event_contains_output(data: str) -> bool:
    """Return whether an SSE payload contains generated completion output.

    The benchmark contract supports either the temporary ``token_id```
    representation or text emitted by a completions-compatible endpoint.

    Args:
        data: JSON payload extracted from an SSE data line.

    Returns:
        Whether the event contains generated output.

    Raises:
        RuntimeError: If the SSE payload is not valid JSON.
    """
    try:
        payload: object = json.loads(data)
    except json.JSONDecodeError as error:
        raise RuntimeError("received an invalid JSON SSE payload.") from error

    if not isinstance(payload, dict):
        return False

    choices: object = payload.get("choices")

    if not isinstance(choices, list) or not choices:
        return False

    choice: object = choices[0]

    if not isinstance(choice, dict):
        return False

    token_id: object = choice.get("token_id")
    if isinstance(token_id, int):
        return True

    text: object = choice.get("text")
    return isinstance(text, str) and bool(text)
