"""Structured results and reporting for LLM serving benchmarks."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from serving.benchmark.conditions import BenchmarkConditions
from serving.benchmark.metrics import (
    AggregateMetrics,
    LatencySummary,
    RequestMetrics,
    compute_aggregate_metrics,
    compute_latency_summary,
)


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Describe the workload used for one benchmark run.

    Attributes:
        request_count: Number of requests submitted.
        prompt_token_ids: Prompt token identifiers shared by requests.
        max_tokens: Maximum output tokens requested per completion.
        max_concurrency: Maximum client requests allowed in flight.
        arrival_mode: Request arrival process, such as fixed or poisson.
        request_rate_per_second: Poisson arrival rate when configured.
        seed: Poisson random seed when configured.
    """

    request_count: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    max_concurrency: int
    arrival_mode: str
    request_rate_per_second: float | None
    seed: int | None


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete result of one LLM serving benchmark run.

    Attributes:
        conditions: Conditions under which the benchmark was executed.
        workload: Workload configuration used for the benchmark.
        aggregate_metrics: Throughput and goodput measurements for the run.
        ttft: Time-to-first-token percentile summary.
        itl: Inter-token-latency percentile summary.
        tpot: Time-per-output-token percentile summary.
        e2e: End-to-end latency percentile summary.
        request_metrics: Per-request measurements retained for later analysis.
    """

    conditions: BenchmarkConditions
    workload: BenchmarkWorkload
    aggregate_metrics: AggregateMetrics
    ttft: LatencySummary
    itl: LatencySummary
    tpot: LatencySummary
    e2e: LatencySummary
    request_metrics: tuple[RequestMetrics, ...]


def build_benchmark_result(
    *,
    conditions: BenchmarkConditions,
    workload: BenchmarkWorkload,
    request_metrics: tuple[RequestMetrics, ...],
    benchmark_duration_ns: int,
) -> BenchmarkResult:
    """Assemble aggregate and percentile results for one benchmark run.

    Args:
        conditions: Conditions under which the benchmark was executed.
        workload: Workload configuration used to generate and execute the requests.
        request_metrics: Per-request measurements collected during the run.
        benchmark_duration_ns: Wall-clock duration of the complete benchmark.

    Returns:
        Complete structured benchmark result.
    """
    aggregate_metrics = compute_aggregate_metrics(
        request_metrics=request_metrics,
        benchmark_duration_ns=benchmark_duration_ns,
        slo=conditions.slo,
    )

    ttft_values = tuple(metrics.ttft_ms for metrics in request_metrics)

    itl_values = tuple(
        latency for metrics in request_metrics for latency in metrics.itl_ms
    )

    tpot_values = tuple(metrics.tpot_ms for metrics in request_metrics)
    e2e_values = tuple(metrics.e2e_ms for metrics in request_metrics)

    return BenchmarkResult(
        conditions=conditions,
        workload=workload,
        aggregate_metrics=aggregate_metrics,
        ttft=compute_latency_summary(ttft_values),
        itl=compute_latency_summary(itl_values),
        tpot=compute_latency_summary(tpot_values),
        e2e=compute_latency_summary(e2e_values),
        request_metrics=request_metrics,
    )


def write_benchmark_result(result: BenchmarkResult, output_path: Path) -> None:
    """Write a benchmark result as formatted JSON.

    Parent directories are created automatically when they do not exist.

    Args:
        result: Complete benchmark result to serialize.
        output_path: Destination JSON file.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
        file.write("\n")


def format_benchmark_summary(result: BenchmarkResult) -> str:
    """Format a human-readable summary of benchmark results.

    Args:
        result: Complete benchmark result to summarize.

    Returns:
        Human-readable latency and throughput summary.
    """
    latency_rows = (
        ("TTFT", result.ttft),
        ("ITL", result.itl),
        ("TPOT", result.tpot),
        ("E2E", result.e2e),
    )

    lines = [
        "Latency (ms)",
        "------------",
        f"{'Metric':<8} {'Samples':>8} {'P50':>10} {'P90':>10} {'P99':>10}",
    ]

    for name, summary in latency_rows:
        lines.append(
            f"{name:<8} "
            f"{summary.sample_count:>8} "
            f"{_format_latency(summary.p50_ms):>10} "
            f"{_format_latency(summary.p90_ms):>10} "
            f"{_format_latency(summary.p99_ms):>10}"
        )

    aggregate = result.aggregate_metrics

    lines.extend(
        [
            "",
            "Throughput",
            "----------",
            (f"Output:   {aggregate.output_throughput_tokens_per_second:.2f} tok/s"),
            (f"Requests: {aggregate.request_throughput_per_second:.2f} req/s"),
            (
                "Goodput:  "
                f"{aggregate.goodput_requests_per_second:.2f} req/s "
                f"({aggregate.slo_pass_count}/{aggregate.request_count} requests)"
            ),
        ]
    )

    return "\n".join(lines)


def _format_latency(value: float | None) -> str:
    """Format an optional latency value for human-readable output."""
    return "N/A" if value is None else f"{value:.2f}"
