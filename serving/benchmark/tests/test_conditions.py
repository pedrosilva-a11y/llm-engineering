"""Tests for benchmark reproducibility conditions."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from serving.benchmark.conditions import (
    capture_benchmark_conditions,
    capture_repository_state,
)
from serving.benchmark.metrics import SLO


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    """Create a temporary Git repository with one committed file."""
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.name", "Benchmark Test")
    _run_git(tmp_path, "config", "user.email", "benchmark@example.com")

    tracked_file = tmp_path / "tracked.txt"
    tracked_file.write_text("initial content\n")

    _run_git(tmp_path, "add", "tracked.txt")
    _run_git(tmp_path, "commit", "-m", "initial commit")

    return tmp_path


def test_capture_repository_state_reads_current_commit(
    git_repository: Path,
) -> None:
    """Capture the commit SHA checked out in the repository."""
    expected_sha = _run_git(git_repository, "rev-parse", "HEAD")

    state = capture_repository_state(repository_path=git_repository)

    assert state.git_sha == expected_sha


def test_capture_repository_state_reports_clean_worktree(
    git_repository: Path,
) -> None:
    """Report a clean repository when no changes are present."""
    state = capture_repository_state(repository_path=git_repository)

    assert state.git_dirty is False


def test_capture_repository_state_reports_modified_tracked_file(
    git_repository: Path,
) -> None:
    """Report a dirty repository when a tracked file is modified."""
    tracked_file = git_repository / "tracked.txt"
    tracked_file.write_text("modified content\n")

    state = capture_repository_state(repository_path=git_repository)

    assert state.git_dirty is True


def test_capture_repository_state_reports_untracked_file(
    git_repository: Path,
) -> None:
    """Report a dirty repository when an untracked file is present."""
    untracked_file = git_repository / "untracked.txt"
    untracked_file.write_text("untracked content\n")

    state = capture_repository_state(repository_path=git_repository)

    assert state.git_dirty is True


def test_capture_repository_state_fails_outside_git_repository(
    tmp_path: Path,
) -> None:
    """Raise an error when repository state cannot be read."""
    with pytest.raises(RuntimeError, match="failed to read repository state with git"):
        capture_repository_state(
            repository_path=tmp_path,
        )


def test_capture_benchmark_conditions_collects_run_metadata(
    git_repository: Path,
) -> None:
    """Capture supplied benchmark metadata and repository state."""
    slo = SLO(max_ttft_ms=100.0, max_tpot_ms=50.0)

    conditions = capture_benchmark_conditions(
        model_name="stub-model",
        hardware="cpu",
        engine_name="custom-engine",
        engine_version="0.1.0",
        quantization=None,
        cache_state="cold",
        slo=slo,
        repository_path=git_repository,
    )

    expected_sha = _run_git(git_repository, "rev-parse", "HEAD")

    assert conditions.model_name == "stub-model"
    assert conditions.hardware == "cpu"
    assert conditions.engine_name == "custom-engine"
    assert conditions.engine_version == "0.1.0"
    assert conditions.quantization is None
    assert conditions.cache_state == "cold"
    assert conditions.slo == slo

    assert conditions.repository.git_sha == expected_sha
    assert conditions.repository.git_dirty is False


def test_capture_benchmark_conditions_records_utc_timestamp(
    git_repository: Path,
) -> None:
    """Capture a parseable timezone-aware UTC timestamp."""
    conditions = capture_benchmark_conditions(
        model_name="stub-model",
        hardware="cpu",
        engine_name="custom-engine",
        engine_version=None,
        quantization=None,
        cache_state="cold",
        slo=SLO(max_ttft_ms=100.0),
        repository_path=git_repository,
    )

    captured_at = datetime.fromisoformat(conditions.captured_at_utc)

    assert captured_at.tzinfo is not None
    assert captured_at.utcoffset() == UTC.utcoffset(captured_at)


def _run_git(repository_path: Path, *arguments: str) -> str:
    """Run Git in a test repository and return standard output."""
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()
