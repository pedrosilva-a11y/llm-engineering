"""Command-line interface for running LLM serving benchmarks."""

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from serving.benchmark.client import (
    BenchmarkClient,
    BenchmarkClientConfiguration,
)
from serving.benchmark.conditions import capture_benchmark_conditions
from serving.benchmark.metrics import SLO
from serving.benchmark.results import (
    BenchmarkResult,
    BenchmarkWorkload,
    build_benchmark_result,
    format_benchmark_summary,
    write_benchmark_result,
)
from serving.benchmark.workload import (
    WorkloadRequest,
    WorkloadSpecification,
    generate_fixed_workload,
    generate_poisson_workload,
)


@dataclass(frozen=True)
class BenchmarkCLIArguments:
    """Validated arguments for one benchmark invocation.

    Attributes:
        endpoint_url: Streaming completions endpoint to benchmark.
        model_name: Model identifier sent to the serving endpoint.
        hardware: Hardware description recorded with the result.
        engine_name: Serving engine name recorded with the result.
        engine_version: Optional serving engine version.
        quantization: Optional model quantization description.
        cache_state: Cache state recorded with the benchmark.
        request_count: Number of requests in the workload.
        prompt_token_ids: Tokenized prompt shared by generated requests.
        max_tokens: Maximum number of output tokens per request.
        max_concurrency: Maximum number of simultaneous client requests.
        timeout_seconds: HTTP timeout for each benchmark request.
        arrival_mode: Workload arrival mode, fixed or poisson.
        request_rate_per_second: Poisson request arrival rate when applicable.
        seed: Random seed used for Poisson workload generation.
        max_ttft_ms: Maximum TTFT allowed by the benchmark SLO.
        max_tpot_ms: Optional maximum TPOT allowed by the benchmark SLO.
        max_e2e_ms: Optional maximum E2E latency allowed by the benchmark SLO.
        output_path: JSON result output path.
    """

    endpoint_url: str
    model_name: str
    hardware: str
    engine_name: str
    engine_version: str | None
    quantization: str | None
    cache_state: str
    request_count: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    max_concurrency: int
    timeout_seconds: float
    arrival_mode: str
    request_rate_per_second: float | None
    seed: int
    max_ttft_ms: float
    max_tpot_ms: float | None
    max_e2e_ms: float | None
    output_path: Path


def parse_args(
    argv: Sequence[str] | None = None,
) -> BenchmarkCLIArguments:
    """Parse benchmark command-line arguments.

    Args:
        argv: Optional argument sequence. Defaults to process command-line
            arguments when omitted.

    Returns:
        Validated benchmark CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run an LLM serving benchmark.",
    )

    parser.add_argument(
        "--endpoint",
        required=True,
        help="Streaming /v1/completions endpoint.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier sent to the endpoint.",
    )
    parser.add_argument(
        "--hardware",
        required=True,
        help="Hardware description recorded in benchmark conditions.",
    )
    parser.add_argument(
        "--engine-name",
        default="custom-engine",
        help="Serving engine name recorded in benchmark conditions.",
    )
    parser.add_argument(
        "--engine-version",
        default=None,
        help="Optional serving engine version.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional quantization description.",
    )
    parser.add_argument(
        "--cache-state",
        default="cold",
        help="Cache state recorded in benchmark conditions.",
    )
    parser.add_argument(
        "--requests",
        dest="request_count",
        type=_positive_int,
        default=20,
        help="Number of benchmark requests.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        nargs="+",
        type=_non_negative_int,
        default=(1, 2, 3),
        help="Prompt token IDs used by every request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=8,
        help="Maximum output tokens per request.",
    )
    parser.add_argument(
        "--concurrency",
        dest="max_concurrency",
        type=_positive_int,
        default=1,
        help="Maximum simultaneous client requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("fixed", "poisson"),
        default="fixed",
        help="Workload request-arrival mode.",
    )
    parser.add_argument(
        "--request-rate",
        dest="request_rate_per_second",
        type=_positive_float,
        default=None,
        help="Poisson request rate in requests per second.",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=0,
        help="Random seed for workload generation.",
    )
    parser.add_argument(
        "--max-ttft-ms",
        type=_positive_float,
        required=True,
        help="Maximum TTFT allowed by the benchmark SLO.",
    )
    parser.add_argument(
        "--max-tpot-ms",
        type=_positive_float,
        default=None,
        help="Optional maximum TPOT allowed by the benchmark SLO.",
    )
    parser.add_argument(
        "--max-e2e-ms",
        type=_positive_float,
        default=None,
        help="Optional maximum E2E latency allowed by the benchmark SLO.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=Path("benchmark_results/result.json"),
        help="Path for the benchmark JSON result.",
    )

    namespace = parser.parse_args(argv)

    if (
        namespace.arrival_mode == "poisson"
        and namespace.request_rate_per_second is None
    ):
        parser.error("--request-rate is required for poisson arrival mode.")

    if (
        namespace.arrival_mode == "fixed"
        and namespace.request_rate_per_second is not None
    ):
        parser.error("--request-rate is only valid for poisson arrival mode.")

    return BenchmarkCLIArguments(
        endpoint_url=namespace.endpoint,
        model_name=namespace.model,
        hardware=namespace.hardware,
        engine_name=namespace.engine_name,
        engine_version=namespace.engine_version,
        quantization=namespace.quantization,
        cache_state=namespace.cache_state,
        request_count=namespace.request_count,
        prompt_token_ids=tuple(namespace.prompt_token_ids),
        max_tokens=namespace.max_tokens,
        max_concurrency=namespace.max_concurrency,
        timeout_seconds=namespace.timeout_seconds,
        arrival_mode=namespace.arrival_mode,
        request_rate_per_second=namespace.request_rate_per_second,
        seed=namespace.seed,
        max_ttft_ms=namespace.max_ttft_ms,
        max_tpot_ms=namespace.max_tpot_ms,
        max_e2e_ms=namespace.max_e2e_ms,
        output_path=namespace.output_path,
    )


async def execute_benchmark(
    arguments: BenchmarkCLIArguments,
) -> BenchmarkResult:
    """Execute one benchmark using parsed CLI arguments.

    Args:
        arguments: Validated benchmark configuration.

    Returns:
        Complete benchmark result including conditions and aggregate metrics.
    """
    slo = SLO(
        max_ttft_ms=arguments.max_ttft_ms,
        max_tpot_ms=arguments.max_tpot_ms,
        max_e2e_ms=arguments.max_e2e_ms,
    )

    conditions = capture_benchmark_conditions(
        model_name=arguments.model_name,
        hardware=arguments.hardware,
        engine_name=arguments.engine_name,
        engine_version=arguments.engine_version,
        quantization=arguments.quantization,
        cache_state=arguments.cache_state,
        slo=slo,
    )

    workload = _build_workload(arguments)

    client = BenchmarkClient(
        configuration=BenchmarkClientConfiguration(
            endpoint_url=arguments.endpoint_url,
            model_name=arguments.model_name,
            max_concurrency=arguments.max_concurrency,
            timeout_seconds=arguments.timeout_seconds,
        )
    )

    if arguments.arrival_mode == "fixed":
        execution = await client.run_fixed_workload(workload)
    else:
        execution = await client.run_scheduled_workload(workload)

    benchmark_workload = BenchmarkWorkload(
        request_count=arguments.request_count,
        prompt_token_ids=arguments.prompt_token_ids,
        max_tokens=arguments.max_tokens,
        max_concurrency=arguments.max_concurrency,
        arrival_mode=arguments.arrival_mode,
        request_rate_per_second=arguments.request_rate_per_second,
        seed=arguments.seed if arguments.arrival_mode == "poisson" else None,
    )

    return build_benchmark_result(
        conditions=conditions,
        workload=benchmark_workload,
        request_metrics=execution.request_metrics,
        benchmark_duration_ns=execution.benchmark_duration_ns,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run a benchmark and emit JSON and terminal summaries."""
    arguments = parse_args(argv)

    result = asyncio.run(
        execute_benchmark(arguments),
    )

    write_benchmark_result(
        result,
        arguments.output_path,
    )

    print(format_benchmark_summary(result))
    print(f"\nResult written to {arguments.output_path}")


def _build_workload(
    arguments: BenchmarkCLIArguments,
) -> tuple[WorkloadRequest, ...]:
    """Generate the workload selected by CLI arguments."""
    specification = WorkloadSpecification(
        request_count=arguments.request_count,
        prompt_token_ids=arguments.prompt_token_ids,
        max_tokens=arguments.max_tokens,
    )

    if arguments.arrival_mode == "fixed":
        return generate_fixed_workload(specification)

    if arguments.arrival_mode == "poisson":
        if arguments.request_rate_per_second is None:
            raise ValueError(
                "request_rate_per_second is required for poisson workloads."
            )

        return generate_poisson_workload(
            specification=specification,
            request_rate_per_second=arguments.request_rate_per_second,
            seed=arguments.seed,
        )

    raise ValueError(
        f"unsupported arrival mode: {arguments.arrival_mode!r}",
    )


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer argument."""
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")

    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer argument."""
    parsed = int(value)

    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative.")

    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive floating-point argument."""
    parsed = float(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")

    return parsed


if __name__ == "__main__":  # pragma: no cover
    main()
