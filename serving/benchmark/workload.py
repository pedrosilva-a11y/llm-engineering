"""Deterministic workload generation for LLM serving benchmarking."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadSpecification:
    """Describe the requests contained in a benchmark workload.

    Attributes:
        request_count: Number of requests generated for the workload.
        prompt_token_ids: Prompt token identifiers shared by generated requests.
        max_tokens: Maximum number of output tokens requested per completion.
        request_id_prefix: Prefix used to generate deterministic request IDs.
    """

    request_count: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    request_id_prefix: str = "request"

    def __post_init__(self) -> None:
        """Validate the workload specification."""
        if self.request_count <= 0:
            raise ValueError("request_count must be greater than zero.")

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty.")

        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError(
                "prompt_token_ids must contain only non-negative token IDs.",
            )

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        if not self.request_id_prefix:
            raise ValueError("request_id_prefix must not be empty.")


@dataclass(frozen=True)
class WorkloadRequest:
    """Describe one scheduled benchmark request.

    Attributes:
        request_id: Deterministic identifier for the request.
        prompt_token_ids: Prompt token identifiers sent to the server.
        max_tokens: Maximum number of output tokens requested.
        arrival_offset_seconds: Time after benchmark start when the request
            becomes eligible for submission.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    arrival_offset_seconds: float

    def __post_init__(self) -> None:
        """Validate the scheduled request."""
        if not self.request_id:
            raise ValueError("request_id must not be empty.")

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty.")

        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError(
                "prompt_token_ids must contain only non-negative token IDs.",
            )

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        if self.arrival_offset_seconds < 0:
            raise ValueError(
                "arrival_offset_seconds must be greater than or equal to zero.",
            )


def generate_fixed_workload(
    specification: WorkloadSpecification,
) -> tuple[WorkloadRequest, ...]:
    """Generate a workload whose requests are all immediately available.

    Fixed-concurrency execution is applied later by the benchmark client. This
    function only generates the requests and assigns each an arrival offset of
    zero.

    Args:
        specification: Workload parameters shared by generated requests.

    Returns:
        Deterministically ordered requests with zero arrival offsets.
    """
    return tuple(
        _build_request(
            specification=specification,
            request_index=request_index,
            arrival_offset_seconds=0.0,
        )
        for request_index in range(specification.request_count)
    )


def generate_poisson_workload(
    *,
    specification: WorkloadSpecification,
    request_rate_per_second: float,
    seed: int,
) -> tuple[WorkloadRequest, ...]:
    """Generate requests with deterministic Poisson arrival times.

    The first request arrives at benchmark time zero. Subsequent inter-arrival
    intervals are sampled from an exponential distribution whose rate is
    ``request_rate_per_second``.

    Args:
        specification: Workload parameters shared by generated requests.
        request_rate_per_second: Mean request arrival rate in requests per second.
        seed: Seed used for deterministic inter-arrival sampling.

    Returns:
        Requests ordered by non-decreasing arrival offsets.

    Raises:
        ValueError: If the request rate is non-positive or the seed is negative.
    """
    if request_rate_per_second <= 0:
        raise ValueError("request_rate_per_second must be greater than zero.")

    if seed < 0:
        raise ValueError("seed must be greater than or equal to zero.")

    generator = random.Random(seed)

    requests: list[WorkloadRequest] = []
    arrival_offset_seconds = 0.0

    for request_index in range(specification.request_count):
        if request_index > 0:
            arrival_offset_seconds += generator.expovariate(request_rate_per_second)

        requests.append(
            _build_request(
                specification=specification,
                request_index=request_index,
                arrival_offset_seconds=arrival_offset_seconds,
            ),
        )

    return tuple(requests)


def _build_request(
    *,
    specification: WorkloadSpecification,
    request_index: int,
    arrival_offset_seconds: float,
) -> WorkloadRequest:
    """Build the deterministic benchmark request."""
    return WorkloadRequest(
        request_id=f"{specification.request_id_prefix}-{request_index:06d}",
        prompt_token_ids=specification.prompt_token_ids,
        max_tokens=specification.max_tokens,
        arrival_offset_seconds=arrival_offset_seconds,
    )
