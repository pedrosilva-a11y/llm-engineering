"""Tests for the LLM serving benchmark command-line interface."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from serving.benchmark.cli import (
    BenchmarkCLIArguments,
    _build_workload,
    execute_benchmark,
    main,
    parse_args,
)
from serving.benchmark.client import BenchmarkClient, BenchmarkExecution
from serving.benchmark.conditions import BenchmarkConditions, RepositoryState
from serving.benchmark.metrics import SLO, RequestMetrics


def test_parse_args_uses_expected_defaults() -> None:
    """Parse required arguments while applying benchmark defaults."""
    arguments = parse_args(
        [
            "--endpoint",
            "http://localhost:8000/v1/completions",
            "--model",
            "stub-model",
            "--hardware",
            "cpu",
            "--max-ttft-ms",
            "1000",
        ]
    )

    assert arguments.endpoint_url == "http://localhost:8000/v1/completions"
    assert arguments.model_name == "stub-model"
    assert arguments.hardware == "cpu"
    assert arguments.engine_name == "custom-engine"
    assert arguments.engine_version is None
    assert arguments.quantization is None
    assert arguments.cache_state == "cold"
    assert arguments.request_count == 20
    assert arguments.prompt_token_ids == (1, 2, 3)
    assert arguments.max_tokens == 8
    assert arguments.max_concurrency == 1
    assert arguments.timeout_seconds == 30.0
    assert arguments.arrival_mode == "fixed"
    assert arguments.request_rate_per_second is None
    assert arguments.seed == 0
    assert arguments.max_ttft_ms == 1000.0
    assert arguments.max_tpot_ms is None
    assert arguments.max_e2e_ms is None
    assert arguments.output_path == Path("benchmark_results/result.json")


def test_parse_args_accepts_poisson_configuration() -> None:
    """Parse an explicitly configured Poisson benchmark."""
    arguments = parse_args(
        [
            "--endpoint",
            "http://localhost:8000/v1/completions",
            "--model",
            "stub-model",
            "--hardware",
            "L4",
            "--engine-name",
            "custom-engine",
            "--engine-version",
            "0.1.0",
            "--quantization",
            "bf16",
            "--cache-state",
            "warm",
            "--requests",
            "50",
            "--prompt-token-ids",
            "10",
            "20",
            "30",
            "--max-tokens",
            "16",
            "--concurrency",
            "8",
            "--timeout-seconds",
            "60",
            "--arrival-mode",
            "poisson",
            "--request-rate",
            "5",
            "--seed",
            "42",
            "--max-ttft-ms",
            "250",
            "--max-tpot-ms",
            "50",
            "--max-e2e-ms",
            "2000",
            "--output",
            "benchmark_results/poisson.json",
        ]
    )

    assert arguments.request_count == 50
    assert arguments.prompt_token_ids == (10, 20, 30)
    assert arguments.max_tokens == 16
    assert arguments.max_concurrency == 8
    assert arguments.timeout_seconds == 60.0
    assert arguments.arrival_mode == "poisson"
    assert arguments.request_rate_per_second == 5.0
    assert arguments.seed == 42
    assert arguments.max_ttft_ms == 250.0
    assert arguments.max_tpot_ms == 50.0
    assert arguments.max_e2e_ms == 2000.0
    assert arguments.output_path == Path("benchmark_results/poisson.json")


def test_parse_args_requires_request_rate_for_poisson() -> None:
    """Reject Poisson mode without a request arrival rate."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--endpoint",
                "http://localhost:8000/v1/completions",
                "--model",
                "stub-model",
                "--hardware",
                "cpu",
                "--arrival-mode",
                "poisson",
                "--max-ttft-ms",
                "1000",
            ]
        )


def test_parse_args_rejects_request_rate_for_fixed_mode() -> None:
    """Reject a Poisson request rate when fixed mode is selected."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--endpoint",
                "http://localhost:8000/v1/completions",
                "--model",
                "stub-model",
                "--hardware",
                "cpu",
                "--request-rate",
                "5",
                "--max-ttft-ms",
                "1000",
            ]
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--requests", "0"),
        ("--max-tokens", "0"),
        ("--concurrency", "0"),
        ("--timeout-seconds", "0"),
        ("--max-ttft-ms", "0"),
        ("--seed", "-1"),
    ),
)
def test_parse_args_rejects_invalid_numeric_values(
    flag: str,
    value: str,
) -> None:
    """Reject invalid positive and non-negative numeric arguments."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--endpoint",
                "http://localhost:8000/v1/completions",
                "--model",
                "stub-model",
                "--hardware",
                "cpu",
                "--max-ttft-ms",
                "1000",
                flag,
                value,
            ]
        )


def test_build_workload_generates_fixed_arrivals() -> None:
    """Generate zero-offset requests for fixed workload mode."""
    arguments = _arguments()

    workload = _build_workload(arguments)

    assert len(workload) == 3
    assert all(request.arrival_offset_seconds == 0.0 for request in workload)


def test_build_workload_generates_poisson_arrivals() -> None:
    """Generate scheduled requests for Poisson workload mode."""
    arguments = replace(
        _arguments(),
        arrival_mode="poisson",
        request_rate_per_second=5.0,
        seed=42,
    )

    workload = _build_workload(arguments)

    assert len(workload) == 3
    assert workload[0].arrival_offset_seconds == 0.0
    assert workload[1].arrival_offset_seconds > 0.0
    assert workload[2].arrival_offset_seconds >= workload[1].arrival_offset_seconds


@pytest.mark.anyio
async def test_execute_benchmark_uses_fixed_client_path() -> None:
    """Execute fixed workloads through the fixed-concurrency client path."""
    arguments = _arguments()
    execution = _execution()
    conditions = _conditions()

    fixed_mock = AsyncMock(return_value=execution)
    scheduled_mock = AsyncMock(return_value=execution)

    with (
        patch.object(
            BenchmarkClient,
            "run_fixed_workload",
            new=fixed_mock,
        ),
        patch.object(
            BenchmarkClient,
            "run_scheduled_workload",
            new=scheduled_mock,
        ),
        patch(
            "serving.benchmark.cli.capture_benchmark_conditions",
            return_value=conditions,
        ),
    ):
        result = await execute_benchmark(arguments)

    fixed_mock.assert_awaited_once()
    scheduled_mock.assert_not_awaited()

    assert result.conditions == conditions
    assert result.request_metrics == execution.request_metrics
    assert result.aggregate_metrics.request_count == 1
    assert result.aggregate_metrics.output_tokens == 2
    assert result.workload.request_count == arguments.request_count
    assert result.workload.prompt_token_ids == arguments.prompt_token_ids
    assert result.workload.max_tokens == arguments.max_tokens
    assert result.workload.max_concurrency == arguments.max_concurrency
    assert result.workload.arrival_mode == "fixed"
    assert result.workload.request_rate_per_second is None
    assert result.workload.seed is None


@pytest.mark.anyio
async def test_execute_benchmark_uses_scheduled_client_path() -> None:
    """Execute Poisson workloads through the scheduled client path."""
    arguments = replace(
        _arguments(),
        arrival_mode="poisson",
        request_rate_per_second=5.0,
    )
    execution = _execution()
    conditions = _conditions()

    fixed_mock = AsyncMock(return_value=execution)
    scheduled_mock = AsyncMock(return_value=execution)

    with (
        patch.object(
            BenchmarkClient,
            "run_fixed_workload",
            new=fixed_mock,
        ),
        patch.object(
            BenchmarkClient,
            "run_scheduled_workload",
            new=scheduled_mock,
        ),
        patch(
            "serving.benchmark.cli.capture_benchmark_conditions",
            return_value=conditions,
        ),
    ):
        result = await execute_benchmark(arguments)

    fixed_mock.assert_not_awaited()
    scheduled_mock.assert_awaited_once()

    assert result.conditions == conditions
    assert result.workload.request_count == arguments.request_count
    assert result.workload.prompt_token_ids == arguments.prompt_token_ids
    assert result.workload.max_tokens == arguments.max_tokens
    assert result.workload.max_concurrency == arguments.max_concurrency
    assert result.workload.arrival_mode == "poisson"
    assert result.workload.request_rate_per_second == 5.0
    assert result.workload.seed == arguments.seed


def test_main_writes_result_and_prints_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write the JSON result and print its human-readable summary."""
    result = object()

    execute_mock = AsyncMock(return_value=result)

    with (
        patch(
            "serving.benchmark.cli.execute_benchmark",
            new=execute_mock,
        ),
        patch(
            "serving.benchmark.cli.write_benchmark_result",
        ) as write_mock,
        patch(
            "serving.benchmark.cli.format_benchmark_summary",
            return_value="benchmark summary",
        ),
    ):
        main(
            [
                "--endpoint",
                "http://localhost:8000/v1/completions",
                "--model",
                "stub-model",
                "--hardware",
                "cpu",
                "--max-ttft-ms",
                "1000",
                "--output",
                "benchmark_results/test.json",
            ]
        )

    write_mock.assert_called_once_with(
        result,
        Path("benchmark_results/test.json"),
    )

    output = capsys.readouterr().out

    assert "benchmark summary" in output
    assert "benchmark_results/test.json" in output


def test_build_workload_rejects_unsupported_mode() -> None:
    """Reject workload modes not supported by the benchmark harness."""
    arguments = replace(
        _arguments(),
        arrival_mode="unsupported",
    )

    with pytest.raises(ValueError, match="unsupported arrival mode"):
        _build_workload(arguments)


def test_build_poisson_workload_requires_request_rate() -> None:
    """Reject direct Poisson configuration without a request rate."""
    arguments = replace(
        _arguments(),
        arrival_mode="poisson",
        request_rate_per_second=None,
    )

    with pytest.raises(
        ValueError,
        match="request_rate_per_second is required",
    ):
        _build_workload(arguments)


def _arguments() -> BenchmarkCLIArguments:
    """Create minimal valid fixed-workload CLI arguments."""
    return BenchmarkCLIArguments(
        endpoint_url="http://localhost:8000/v1/completions",
        model_name="stub-model",
        hardware="cpu",
        engine_name="custom-engine",
        engine_version=None,
        quantization=None,
        cache_state="cold",
        request_count=3,
        prompt_token_ids=(1, 2, 3),
        max_tokens=2,
        max_concurrency=2,
        timeout_seconds=30.0,
        arrival_mode="fixed",
        request_rate_per_second=None,
        seed=0,
        max_ttft_ms=1000.0,
        max_tpot_ms=None,
        max_e2e_ms=None,
        output_path=Path("benchmark_results/test.json"),
    )


def _execution() -> BenchmarkExecution:
    """Create deterministic benchmark execution measurements."""
    return BenchmarkExecution(
        request_metrics=(
            RequestMetrics(
                ttft_ms=10.0,
                itl_ms=(5.0,),
                tpot_ms=5.0,
                e2e_ms=15.0,
                output_tokens=2,
            ),
        ),
        benchmark_duration_ns=1_000_000_000,
    )


def _conditions() -> BenchmarkConditions:
    """Create deterministic benchmark conditions."""
    return BenchmarkConditions(
        model_name="stub-model",
        hardware="cpu",
        engine_name="custom-engine",
        engine_version=None,
        quantization=None,
        cache_state="cold",
        slo=SLO(max_ttft_ms=1000.0),
        captured_at_utc="2026-09-10T18:00:00+00:00",
        repository=RepositoryState(
            git_sha="abcdef123456",
            git_dirty=False,
        ),
    )
