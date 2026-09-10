"""Tests for structured benchmark results and reporting."""

import json
from pathlib import Path

from serving.benchmark.conditions import BenchmarkConditions, RepositoryState
from serving.benchmark.metrics import SLO, RequestMetrics
from serving.benchmark.results import (
    BenchmarkWorkload,
    build_benchmark_result,
    format_benchmark_summary,
    write_benchmark_result,
)


def test_build_benchmark_result_assembles_metrics() -> None:
    """Assemble aggregate metrics and latency distributions correctly."""
    conditions = _benchmark_conditions()
    request_metrics = (
        RequestMetrics(
            ttft_ms=100.0,
            itl_ms=(20.0, 30.0),
            tpot_ms=25.0,
            e2e_ms=150.0,
            output_tokens=3,
        ),
        RequestMetrics(
            ttft_ms=200.0,
            itl_ms=(40.0,),
            tpot_ms=40.0,
            e2e_ms=240.0,
            output_tokens=2,
        ),
        RequestMetrics(
            ttft_ms=300.0,
            itl_ms=(),
            tpot_ms=None,
            e2e_ms=300.0,
            output_tokens=1,
        ),
    )

    result = build_benchmark_result(
        conditions=conditions,
        workload=_benchmark_workload(),
        request_metrics=request_metrics,
        benchmark_duration_ns=2_000_000_000,
    )

    assert result.conditions == conditions
    assert result.request_metrics == request_metrics

    assert result.workload == _benchmark_workload()
    assert result.workload.request_count == 3
    assert result.workload.max_concurrency == 2
    assert result.workload.arrival_mode == "fixed"

    assert result.ttft.sample_count == 3
    assert result.ttft.p50_ms == 200.0

    assert result.itl.sample_count == 3
    assert result.itl.p50_ms == 30.0

    assert result.tpot.sample_count == 2
    assert result.tpot.p50_ms == 32.5

    assert result.e2e.sample_count == 3
    assert result.e2e.p50_ms == 240.0

    assert result.aggregate_metrics.request_count == 3
    assert result.aggregate_metrics.output_tokens == 6
    assert result.aggregate_metrics.slo_pass_count == 2
    assert result.aggregate_metrics.output_throughput_tokens_per_second == 3.0
    assert result.aggregate_metrics.request_throughput_per_second == 1.5
    assert result.aggregate_metrics.goodput_requests_per_second == 1.0


def test_write_benchmark_result_serializes_json(tmp_path: Path) -> None:
    """Write nested benchmark results as formatted JSON."""
    result = build_benchmark_result(
        conditions=_benchmark_conditions(),
        workload=_benchmark_workload(request_count=1),
        request_metrics=(
            RequestMetrics(
                ttft_ms=100.0,
                itl_ms=(20.0,),
                tpot_ms=20.0,
                e2e_ms=120.0,
                output_tokens=2,
            ),
        ),
        benchmark_duration_ns=1_000_000_000,
    )

    output_path = tmp_path / "nested" / "benchmark.json"

    write_benchmark_result(result=result, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert data["conditions"]["model_name"] == "stub-model"
    assert data["conditions"]["repository"]["git_sha"] == "abc123"
    assert data["aggregate_metrics"]["request_count"] == 1
    assert data["ttft"]["p50_ms"] == 100.0
    assert data["request_metrics"][0]["output_tokens"] == 2
    assert data["workload"]["request_count"] == 1
    assert data["workload"]["max_concurrency"] == 2
    assert data["workload"]["arrival_mode"] == "fixed"
    assert data["workload"]["request_rate_per_second"] is None
    assert data["workload"]["seed"] is None


def test_format_benchmark_summary_reports_latency_and_throughput() -> None:
    """Format latency percentiles and throughput for terminal output."""
    result = build_benchmark_result(
        conditions=_benchmark_conditions(),
        workload=_benchmark_workload(request_count=1),
        request_metrics=(
            RequestMetrics(
                ttft_ms=100.0,
                itl_ms=(),
                tpot_ms=None,
                e2e_ms=100.0,
                output_tokens=1,
            ),
        ),
        benchmark_duration_ns=1_000_000_000,
    )

    summary = format_benchmark_summary(result)

    assert "Latency (ms)" in summary
    assert "TTFT" in summary
    assert "ITL" in summary
    assert "TPOT" in summary
    assert "E2E" in summary
    assert "100.00" in summary
    assert "N/A" in summary
    assert "Output:   1.00 tok/s" in summary
    assert "Requests: 1.00 req/s" in summary
    assert "Goodput:  1.00 req/s (1/1 requests)" in summary


def _benchmark_conditions() -> BenchmarkConditions:
    """Create deterministic benchmark conditions for result tests."""
    return BenchmarkConditions(
        model_name="stub-model",
        hardware="cpu",
        engine_name="custom-engine",
        engine_version="0.1.0",
        quantization=None,
        cache_state="cold",
        slo=SLO(max_ttft_ms=250.0),
        captured_at_utc="2026-09-09T12:00:00+00:00",
        repository=RepositoryState(
            git_sha="abc123",
            git_dirty=False,
        ),
    )


def _benchmark_workload(
    *,
    request_count: int = 3,
) -> BenchmarkWorkload:
    """Create deterministic workload configuration for result tests."""
    return BenchmarkWorkload(
        request_count=request_count,
        prompt_token_ids=(1, 2, 3),
        max_tokens=8,
        max_concurrency=2,
        arrival_mode="fixed",
        request_rate_per_second=None,
        seed=None,
    )
