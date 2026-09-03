"""Tests for cost-model report orchestration."""

from pathlib import Path

import pytest

from serving.cost_models.main import main


def test_main_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate a report at the configured output path."""
    output_path = tmp_path / "predictions.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "cost-model",
            "--model",
            "qwen2.5-1.5b",
            "--hardware",
            "L4",
            "--output",
            str(output_path),
        ],
    )

    main()

    rendered = output_path.read_text(encoding="utf-8")

    assert "# LLM Serving Cost-Model Predictions" in rendered
    assert "- Model: qwen2.5-1.5b" in rendered
    assert "- Hardware: L4" in rendered
    assert "## Memory Estimates" in rendered
    assert "## Decode Estimates" in rendered
    assert "## Prefill Estimates" in rendered
