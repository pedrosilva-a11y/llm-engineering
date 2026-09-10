"""Tests for deterministic benchmark workload generation."""

from itertools import pairwise

import pytest

from serving.benchmark.workload import (
    WorkloadRequest,
    WorkloadSpecification,
    generate_fixed_workload,
    generate_poisson_workload,
)


@pytest.mark.parametrize(
    (
        "field_name",
        "request_count",
        "prompt_token_ids",
        "max_tokens",
        "request_id_prefix",
    ),
    (
        ("request_count", 0, (1, 2), 4, "request"),
        ("prompt_token_ids", 2, (), 4, "request"),
        ("prompt_token_ids", 2, (1, -1), 4, "request"),
        ("max_tokens", 2, (1, 2), 0, "request"),
        ("request_id_prefix", 2, (1, 2), 4, ""),
    ),
)
def test_workload_specification_rejects_invalid_values(
    field_name: str,
    request_count: int,
    prompt_token_ids: tuple[int, ...],
    max_tokens: int,
    request_id_prefix: str,
) -> None:
    """Reject one invalid workload field while remaining fields are valid."""
    with pytest.raises(ValueError, match=field_name):
        WorkloadSpecification(
            request_count=request_count,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            request_id_prefix=request_id_prefix,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "request_id",
        "prompt_token_ids",
        "max_tokens",
        "arrival_offset_seconds",
    ),
    (
        ("request_id", "", (1, 2), 4, 0.0),
        ("prompt_token_ids", "request-1", (), 4, 0.0),
        ("prompt_token_ids", "request-1", (1, -1), 4, 0.0),
        ("max_tokens", "request-1", (1, 2), 0, 0.0),
        ("arrival_offset_seconds", "request-1", (1, 2), 4, -0.1),
    ),
)
def test_workload_request_rejects_invalid_values(
    field_name: str,
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    max_tokens: int,
    arrival_offset_seconds: float,
) -> None:
    """Reject one invalid scheduled-request field."""
    with pytest.raises(ValueError, match=field_name):
        WorkloadRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            arrival_offset_seconds=arrival_offset_seconds,
        )


def test_generate_fixed_workload_builds_immediately_available_requests() -> None:
    """Generate deterministic requests whose arrival offsets are all zero."""
    specification = WorkloadSpecification(
        request_count=3,
        prompt_token_ids=(10, 20, 30),
        max_tokens=4,
        request_id_prefix="benchmark",
    )

    requests = generate_fixed_workload(specification)

    assert len(requests) == 3

    assert tuple(request.request_id for request in requests) == (
        "benchmark-000000",
        "benchmark-000001",
        "benchmark-000002",
    )

    assert all(request.prompt_token_ids == (10, 20, 30) for request in requests)
    assert all(request.max_tokens == 4 for request in requests)
    assert all(request.arrival_offset_seconds == 0.0 for request in requests)


def test_generate_poisson_workload_is_reproducible() -> None:
    """Generate identical Poisson arrivals from the same random seed."""
    specification = WorkloadSpecification(
        request_count=10,
        prompt_token_ids=(1, 2),
        max_tokens=4,
    )

    first = generate_poisson_workload(
        specification=specification,
        request_rate_per_second=10.0,
        seed=42,
    )

    second = generate_poisson_workload(
        specification=specification,
        request_rate_per_second=10.0,
        seed=42,
    )

    assert first == second


def test_generate_poisson_workload_starts_at_zero_and_increases() -> None:
    """Start the first request at zero and accumulate later arrival times."""
    specification = WorkloadSpecification(
        request_count=5,
        prompt_token_ids=(1,),
        max_tokens=2,
    )

    requests = generate_poisson_workload(
        specification=specification,
        request_rate_per_second=5.0,
        seed=42,
    )

    arrival_offsets = tuple(request.arrival_offset_seconds for request in requests)

    assert arrival_offsets[0] == 0.0
    assert all(current >= previous for previous, current in pairwise(arrival_offsets))


def test_generate_poisson_workload_matches_expected_average_interval() -> None:
    """Approximate the expected exponential inter-arrival interval."""
    request_rate_per_second = 10.0

    specification = WorkloadSpecification(
        request_count=10_000,
        prompt_token_ids=(1,),
        max_tokens=1,
    )

    requests = generate_poisson_workload(
        specification=specification,
        request_rate_per_second=request_rate_per_second,
        seed=42,
    )

    arrival_offsets = tuple(request.arrival_offset_seconds for request in requests)

    intervals = tuple(
        current - previous for previous, current in pairwise(arrival_offsets)
    )

    mean_interval = sum(intervals) / len(intervals)

    assert mean_interval == pytest.approx(
        1.0 / request_rate_per_second,
        rel=0.05,
    )


@pytest.mark.parametrize(
    ("request_rate_per_second", "seed", "expected_message"),
    (
        (0.0, 42, "request_rate_per_second"),
        (-1.0, 42, "request_rate_per_second"),
        (10.0, -1, "seed"),
    ),
)
def test_generate_poisson_workload_rejects_invalid_parameters(
    request_rate_per_second: float,
    seed: int,
    expected_message: str,
) -> None:
    """Reject invalid Poisson arrival-generation parameters."""
    specification = WorkloadSpecification(
        request_count=2,
        prompt_token_ids=(1,),
        max_tokens=1,
    )

    with pytest.raises(ValueError, match=expected_message):
        generate_poisson_workload(
            specification=specification,
            request_rate_per_second=request_rate_per_second,
            seed=seed,
        )
