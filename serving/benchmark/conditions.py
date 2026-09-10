"""Reproducibility conditions for LLM serving benchmarks."""

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from serving.benchmark.metrics import SLO


@dataclass(frozen=True)
class RepositoryState:
    """Identify the repository state used for a benchmark run.

    Attributes:
        git_sha: Commit SHA checked out when the benchmark was executed.
        git_dirty: Whether tracked or untracked working-tree changes were present.
    """

    git_sha: str
    git_dirty: bool


@dataclass(frozen=True)
class BenchmarkConditions:
    """Describe the stable conditions associated with one benchmark run.

    Workload-specific fields are recorded separately in BenchmarkWorkload.

    Attributes:
        model_name: Model identifier used by the serving system.
        hardware: Human-readable hardware identifier.
        engine_name: Serving engine being benchmarked.
        engine_version: Engine version when one is available.
        quantization: Quantization mode, or None when weights are unquantized.
        cache_state: Cache condition used for the run, such as warm or cold.
        slo: Latency objectives used to compute benchmark goodput.
        captured_at_utc: UTC timestamp identifying when conditions were captured.
        repository: Git repository state associated with the benchmark.
    """

    model_name: str
    hardware: str
    engine_name: str
    engine_version: str | None
    quantization: str | None
    cache_state: str
    slo: SLO
    captured_at_utc: str
    repository: RepositoryState


def capture_benchmark_conditions(
    *,
    model_name: str,
    hardware: str,
    engine_name: str,
    engine_version: str | None,
    quantization: str | None,
    cache_state: str,
    slo: SLO,
    repository_path: Path | None = None,
) -> BenchmarkConditions:
    """Capture stable conditions for one benchmark run.

    Args:
        model_name: Model identifier used by the serving system.
        hardware: Human-readable hardware identifier.
        engine_name: Serving engine being benchmarked.
        engine_version: Engine version when one is available.
        quantization: Quantization mode, or None when weights are unquantized.
        cache_state: Cache condition used for the run.
        slo: Latency objectives used to compute goodput.
        repository_path: Repository directory used to capture Git state.

    Returns:
        Stable benchmark conditions with repository and timestamp metadata.
    """
    return BenchmarkConditions(
        model_name=model_name,
        hardware=hardware,
        engine_name=engine_name,
        engine_version=engine_version,
        quantization=quantization,
        cache_state=cache_state,
        slo=slo,
        captured_at_utc=_current_utc_timestamp(),
        repository=capture_repository_state(repository_path=repository_path),
    )


def capture_repository_state(repository_path: Path | None = None) -> RepositoryState:
    """Capture the Git state associated with a benchmark run.

    Args:
        repository_path: Repository directory in which Git commands are executed.
            When omitted, commands run from the current working directory.

    Returns:
        Commit SHA and dirty-worktree state.

    Raises:
        RuntimeError: If the repository state cannot be read with Git.
    """
    git_sha = _run_git_command(
        arguments=("rev-parse", "HEAD"),
        repository_path=repository_path,
    )

    status = _run_git_command(
        arguments=("status", "--porcelain"),
        repository_path=repository_path,
    )

    return RepositoryState(git_sha=git_sha, git_dirty=bool(status))


def _run_git_command(arguments: tuple[str, ...], repository_path: Path | None) -> str:
    """Run a Git command and return its stripped standard output.

    Args:
        arguments: Git command arguments excluding the ``git`` executable.
        repository_path: Directory in which to execute the Git command.

    Returns:
        Stripped standard output produced by Git.

    Raises:
        RuntimeError: If the Git command cannot be executed successfully.
    """
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as error:
        raise RuntimeError(
            f"failed to read repository state with git {' '.join(arguments)}."
        ) from error

    return result.stdout.strip()


def _current_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()
