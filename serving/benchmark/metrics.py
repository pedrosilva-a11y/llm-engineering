"""Latency metrics for the LLM serving benchmark.

Raw timing values are represented as monotonic nanosecond timestamps. Reported
latencies are expressed in milliseconds.

Definitions:

    TTFT = first output arrival - request start

    ITL = gaps between consecutive output-token arrivals

    TPOT = time from first to final output token divided by the number of
    post-first-token intervals. TPOT is undefined for a one-token response.

    E2E = request completion - request start

    Percentiles use linear interpolation at position ``(n - 1) * q`` over
    ascending values, where ``q`` is the requested quantile in [0, 1].

    Latency summaries exclude undefined values from percentile computation and
    report the number of contributing samples. If no samples are defined, the
    percentiles are undefined.

    Output throughput = total generated output tokens divided by the benchmark
    wall-clock duration.

    Request throughput = completed requests divided by benchmark wall-clock
    duration.

    Goodput = requests satisfying all applicable SLO thresholds divided by benchmark
    wall-clock duration. An undefined TPOT for a single-token response vacuously
    satisfies a configured TPOT objective.
"""

from dataclasses import dataclass

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class RequestMetrics:
    """Latency measurements for one completed benchmark request.

    Attributes:
        ttft_ms: Time from request start to the first output token.
        itl_ms: Latencies between consecutive output tokens.
        tpot_ms: Average latency between output tokens after the first token, or None
            when the request produced only one output token.
        e2e_ms: End-to-end request latency.
        output_tokens: Number of generated output tokens.
    """

    ttft_ms: float
    itl_ms: tuple[float, ...]
    tpot_ms: float | None
    e2e_ms: float
    output_tokens: int


@dataclass(frozen=True)
class LatencySummary:
    """Percentile summary for a latency metric.

    Attributes:
        sample_count: Number of defined samples used in the summary.
        p50_ms: 50th percentile latency, or None when no samples are defined.
        p90_ms: 90th percentile latency, or None when no samples are defined.
        p99_ms: 99th percentile latency, or None when no samples are defined.
    """

    sample_count: int
    p50_ms: float | None
    p90_ms: float | None
    p99_ms: float | None


@dataclass(frozen=True)
class SLO:
    """Latency objectives used to determine benchmark goodput.

    A request satisfies the SLO when every configured and applicable latency objective
    is met. TPOT is not applicable to a single-token response and therefore does not
    cause that request to fail a configured TPOT objective.

    Attributes:
        max_ttft_ms: Maximum allowed time to first token in milliseconds.
        max_tpot_ms: Maximum allowed time per output token in milliseconds.
        max_e2e_ms: Maximum allowed end-to-end latency in milliseconds.
    """

    max_ttft_ms: float | None = None
    max_tpot_ms: float | None = None
    max_e2e_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate configured SLO thresholds.

        Raises:
            ValueError: If no threshold is configured or if a configured threshold
                is not positive.
        """
        thresholds = (
            ("max_ttft_ms", self.max_ttft_ms),
            ("max_tpot_ms", self.max_tpot_ms),
            ("max_e2e_ms", self.max_e2e_ms),
        )

        if all(value is None for _, value in thresholds):
            raise ValueError("at least one SLO threshold must be configured.")

        for name, value in thresholds:
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero.")


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate throughput and goodput measurements for one benchmark run.

    Attributes:
        request_count: Number of completed requests.
        output_tokens: Total number of generated output tokens.
        slo_pass_count: Number of requests satisfying the configured SLO.
        output_throughput_tokens_per_second: Generated output tokens per second.
        request_throughput_per_second: Completed requests per second.
        goodput_requests_per_second: SLO-satisfying requests per second.
    """

    request_count: int
    output_tokens: int
    slo_pass_count: int
    output_throughput_tokens_per_second: float
    request_throughput_per_second: float
    goodput_requests_per_second: float


def compute_request_metrics(
    request_start_ns: int,
    token_arrival_ns: tuple[int, ...],
    request_end_ns: int,
) -> RequestMetrics:
    """Compute latency metrics from recorded request timestamps.

    Args:
        request_start_ns: Monotonic timestamp immediately before request send.
        token_arrival_ns: Monotonic arrival timestamps for generated tokens.
        request_end_ns: Monotonic timestamp when the request completes.

    Returns:
        Latency measurements for the completed request.
    """
    first_token_ns = token_arrival_ns[0]

    ttft_ms = _to_milliseconds(first_token_ns - request_start_ns)

    itl_ms = tuple(
        _to_milliseconds(current_ns - previous_ns)
        for previous_ns, current_ns in zip(
            token_arrival_ns,
            token_arrival_ns[1:],
        )
    )

    output_tokens = len(token_arrival_ns)

    tpot_ms = (
        _to_milliseconds((token_arrival_ns[-1] - first_token_ns) / (output_tokens - 1))
        if output_tokens > 1
        else None
    )

    e2e_ms = _to_milliseconds(request_end_ns - request_start_ns)

    return RequestMetrics(
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
    )


def compute_percentile(values: tuple[float, ...], quantile: float) -> float:
    """Compute a percentile using linear interpolation.

    Args:
        values: Metric samples to aggregate.
        quantile: Requested quantile in the interval [0, 1].

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If values is empty or quantile is outside [0, 1].
    """
    if not values:
        raise ValueError("values must not be empty.")

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one.")

    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * quantile

    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    interpolation = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + interpolation * (upper_value - lower_value)


def compute_latency_summary(values: tuple[float | None, ...]) -> LatencySummary:
    """Summarize defined latency samples with P50, P90, and P99.

    Undefined samples are excluded from percentile computation. If no defined
    samples remain, the percentile values are None.

    Args:
        values: Latency samples in milliseconds, including optional undefined
            values.

    Returns:
        Percentile summary and number of defined samples.
    """
    defined_values = tuple(value for value in values if value is not None)

    if not defined_values:
        return LatencySummary(
            sample_count=0,
            p50_ms=None,
            p90_ms=None,
            p99_ms=None,
        )

    return LatencySummary(
        sample_count=len(defined_values),
        p50_ms=compute_percentile(defined_values, 0.50),
        p90_ms=compute_percentile(defined_values, 0.90),
        p99_ms=compute_percentile(defined_values, 0.99),
    )


def request_meets_slo(metrics: RequestMetrics, slo: SLO) -> bool:
    """Return whether a request satisfies all applicable SLO objectives.

    Args:
        metrics: Completed request latency measurements.
        slo: Configured latency objectives.

    Returns:
        True when every configured and applicable objective is satisfied.
    """
    if slo.max_ttft_ms is not None and metrics.ttft_ms > slo.max_ttft_ms:
        return False

    if (
        slo.max_tpot_ms is not None
        and metrics.tpot_ms is not None
        and metrics.tpot_ms > slo.max_tpot_ms
    ):
        return False

    if slo.max_e2e_ms is not None and metrics.e2e_ms > slo.max_e2e_ms:
        return False

    return True


def compute_aggregate_metrics(
    request_metrics: tuple[RequestMetrics, ...],
    benchmark_duration_ns: int,
    slo: SLO,
) -> AggregateMetrics:
    """Compute throughput and goodput across a benchmark run.

    Args:
        request_metrics: Metrics for completed benchmark requests.
        benchmark_duration_ns: Wall-clock duration of the complete benchmark run.
        slo: Latency objectives used to determine goodput.

    Returns:
        Aggregate throughput, counts, and goodput measurements.

    Raises:
        ValueError: If benchmark_duration_ns is not positive.
    """
    if benchmark_duration_ns <= 0:
        raise ValueError("benchmark_duration_ns must be greater than zero.")

    duration_seconds = _to_seconds(benchmark_duration_ns)

    request_count = len(request_metrics)
    output_tokens = sum(metrics.output_tokens for metrics in request_metrics)
    slo_pass_count = sum(
        request_meets_slo(metrics=metrics, slo=slo) for metrics in request_metrics
    )

    return AggregateMetrics(
        request_count=request_count,
        output_tokens=output_tokens,
        slo_pass_count=slo_pass_count,
        output_throughput_tokens_per_second=output_tokens / duration_seconds,
        request_throughput_per_second=request_count / duration_seconds,
        goodput_requests_per_second=slo_pass_count / duration_seconds,
    )


def _to_milliseconds(nanoseconds: float) -> float:
    """Convert nanoseconds to milliseconds."""
    return nanoseconds / _NANOSECONDS_PER_MILLISECOND


def _to_seconds(nanoseconds: float) -> float:
    """Convert nanoseconds to seconds."""
    return nanoseconds / _NANOSECONDS_PER_SECOND
