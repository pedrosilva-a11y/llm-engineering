"""Tests for benchmark latency metrics."""

import pytest

from serving.benchmark.metrics import (
    SLO,
    RequestMetrics,
    compute_aggregate_metrics,
    compute_latency_summary,
    compute_percentile,
    compute_request_metrics,
    request_meets_slo,
)


def test_compute_request_metrics_from_token_timestamps() -> None:
    """Compute TTFT, ITL, TPOT, and end-to-end latency."""
    request_start_ns = 0
    token_arrival_ns = (
        100_000_000,
        130_000_000,
        160_000_000,
        190_000_000,
    )
    request_end_ns = 190_000_000

    metrics = compute_request_metrics(
        request_start_ns=request_start_ns,
        token_arrival_ns=token_arrival_ns,
        request_end_ns=request_end_ns,
    )

    assert metrics.ttft_ms == 100.0
    assert metrics.itl_ms == (30.0, 30.0, 30.0)
    assert metrics.tpot_ms == 30.0
    assert metrics.e2e_ms == 190.0
    assert metrics.output_tokens == 4


def test_single_output_token_has_no_tpot() -> None:
    """Leave TPOT undefined when no post-first-token interval exists."""
    metrics = compute_request_metrics(
        request_start_ns=0,
        token_arrival_ns=(100_000_000,),
        request_end_ns=100_000_000,
    )

    assert metrics.ttft_ms == 100.0
    assert metrics.itl_ms == ()
    assert metrics.tpot_ms is None
    assert metrics.e2e_ms == 100.0
    assert metrics.output_tokens == 1


# Percentiles


def test_compute_percentile_at_exact_position() -> None:
    """Return the sample value when the percentile lands on an exact index."""
    values = (10.0, 20.0, 30.0, 40.0, 50.0)

    percentile = compute_percentile(values=values, quantile=0.5)

    assert percentile == 30.0


def test_compute_percentile_interpolates_between_positions() -> None:
    """Linearly interpolate when the percentile falls between sample indices."""
    values = (10.0, 20.0, 30.0, 40.0, 50.0)

    percentile = compute_percentile(values=values, quantile=0.9)

    assert percentile == 46.0


def test_compute_percentile_sorts_samples_before_interpolation() -> None:
    """Compute percentiles independently of input sample ordering."""
    values = (50.0, 10.0, 40.0, 20.0, 30.0)

    percentile = compute_percentile(values=values, quantile=0.5)

    assert percentile == 30.0


def test_compute_percentile_rejects_empty_values() -> None:
    """Reject percentile computation without metric samples."""
    with pytest.raises(ValueError, match="values must not be empty"):
        compute_percentile(values=(), quantile=0.5)


@pytest.mark.parametrize("quantile", (-0.1, 1.1))
def test_compute_percentile_rejects_invalid_quantile(quantile: float) -> None:
    """Reject quantiles outside the interval from zero to one."""
    with pytest.raises(ValueError, match="quantile must be between zero and one"):
        compute_percentile(values=(10.0, 20.0, 30.0), quantile=quantile)


# Latency summaries


def test_compute_latency_summary_reports_percentiles() -> None:
    """Compute P50, P90, and P99 from defined latency samples."""
    values = (10.0, 20.0, 30.0, 40.0, 50.0)

    summary = compute_latency_summary(values=values)

    assert summary.sample_count == 5
    assert summary.p50_ms == pytest.approx(30.0)
    assert summary.p90_ms == pytest.approx(46.0)
    assert summary.p99_ms == pytest.approx(49.6)


def test_compute_latency_summary_excludes_undefined_samples() -> None:
    """Exclude undefined latency values from percentile computation."""
    values = (20.0, None, 25.0, None, 30.0)

    summary = compute_latency_summary(values)

    assert summary.sample_count == 3
    assert summary.p50_ms == pytest.approx(25.0)
    assert summary.p90_ms == pytest.approx(29.0)
    assert summary.p99_ms == pytest.approx(29.9)


def test_compute_latency_summary_without_defined_samples() -> None:
    """Return undefined percentiles when no latency samples are available."""
    values = (None, None)

    summary = compute_latency_summary(values=values)

    assert summary.sample_count == 0
    assert summary.p50_ms is None
    assert summary.p90_ms is None
    assert summary.p99_ms is None


# SLO


def test_slo_requires_at_least_one_threshold() -> None:
    """Reject an SLO without any configured latency objective."""
    with pytest.raises(
        ValueError,
        match="at least one SLO threshold must be configured",
    ):
        SLO()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("max_ttft_ms", 0.0),
        ("max_ttft_ms", -1.0),
        ("max_tpot_ms", 0.0),
        ("max_tpot_ms", -1.0),
        ("max_e2e_ms", 0.0),
        ("max_e2e_ms", -1.0),
    ),
)
def test_slo_rejects_non_positive_threshold(field_name: str, value: float) -> None:
    """Reject configured SLO thresholds that are not positive."""
    with pytest.raises(ValueError, match=f"{field_name} must be greater than zero"):
        SLO(**{field_name: value})


def test_request_meets_all_configured_slo_thresholds() -> None:
    """Accept a request that satisfies every configured objective."""
    metrics = RequestMetrics(
        ttft_ms=50.0,
        itl_ms=(20.0, 20.0),
        tpot_ms=20.0,
        e2e_ms=90.0,
        output_tokens=3,
    )
    slo = SLO(
        max_ttft_ms=50.0,
        max_tpot_ms=20.0,
        max_e2e_ms=90.0,
    )

    assert request_meets_slo(metrics=metrics, slo=slo) is True


def test_request_fails_when_any_slo_threshold_is_exceeded() -> None:
    """Reject a request when any configured latency objective is exceeded."""
    metrics = RequestMetrics(
        ttft_ms=50.0,
        itl_ms=(20.0, 20.0),
        tpot_ms=20.0,
        e2e_ms=90.0,
        output_tokens=3,
    )

    assert request_meets_slo(metrics=metrics, slo=SLO(max_ttft_ms=49.0)) is False

    assert request_meets_slo(metrics=metrics, slo=SLO(max_tpot_ms=19.0)) is False

    assert request_meets_slo(metrics=metrics, slo=SLO(max_e2e_ms=89.0)) is False


def test_undefined_tpot_vacuously_satisfies_tpot_slo() -> None:
    """Ignore the TPOT objective when a request has no defined TPOT."""
    metrics = RequestMetrics(
        ttft_ms=40.0,
        itl_ms=(),
        tpot_ms=None,
        e2e_ms=40.0,
        output_tokens=1,
    )

    slo = SLO(max_tpot_ms=1.0)

    assert request_meets_slo(metrics=metrics, slo=slo) is True


# Aggregate metrics


def test_compute_aggregate_metrics_uses_benchmark_wall_duration() -> None:
    """Compute throughput and goodput from the complete benchmark duration."""
    request_metrics = (
        RequestMetrics(
            ttft_ms=40.0,
            itl_ms=(20.0, 20.0, 20.0),
            tpot_ms=20.0,
            e2e_ms=1_500.0,
            output_tokens=4,
        ),
        RequestMetrics(
            ttft_ms=80.0,
            itl_ms=(30.0,),
            tpot_ms=30.0,
            e2e_ms=1_500.0,
            output_tokens=2,
        ),
    )

    metrics = compute_aggregate_metrics(
        request_metrics=request_metrics,
        benchmark_duration_ns=2_000_000_000,
        slo=SLO(max_ttft_ms=50.0),
    )

    assert metrics.request_count == 2
    assert metrics.output_tokens == 6
    assert metrics.slo_pass_count == 1

    assert metrics.output_throughput_tokens_per_second == pytest.approx(3.0)
    assert metrics.request_throughput_per_second == pytest.approx(1.0)
    assert metrics.goodput_requests_per_second == pytest.approx(0.5)


@pytest.mark.parametrize("benchmark_duration_ns", (0, -1))
def test_compute_aggregate_metrics_rejects_non_positive_duration(
    benchmark_duration_ns: int,
) -> None:
    """Reject aggregate computation without positive benchmark duration."""
    with pytest.raises(
        ValueError,
        match="benchmark_duration_ns must be greater than zero",
    ):
        compute_aggregate_metrics(
            request_metrics=(),
            benchmark_duration_ns=benchmark_duration_ns,
            slo=SLO(max_ttft_ms=100.0),
        )
