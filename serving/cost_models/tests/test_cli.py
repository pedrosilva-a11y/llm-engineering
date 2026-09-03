"""Tests for cost-model command-line parsing."""

from pathlib import Path

import pytest

from serving.cost_models.cli import parse_args


def test_parse_args_required_values_and_defaults() -> None:
    """Parse required arguments while preserving report defaults."""
    arguments = parse_args(
        [
            "--model",
            "qwen2.5-1.5b",
            "--hardware",
            "L4",
        ]
    )

    assert arguments.model == "qwen2.5-1.5b"
    assert arguments.hardware == "L4"
    assert arguments.output == Path("serving/cost_models/outputs/analytical_report.md")
    assert arguments.bytes_per_value == 2
    assert arguments.context_lengths == (512, 1_024, 2_048, 4_096, 8_192)
    assert arguments.batch_sizes == (1, 8, 32, 128)
    assert arguments.prompt_lengths == (128, 256, 384, 512, 1_024, 2_048)
    assert arguments.prefill_batch_size == 1


def test_parse_args_custom_values() -> None:
    """Parse explicitly supplied report-generation arguments."""
    arguments = parse_args(
        [
            "--model",
            "llama-3-8b",
            "--hardware",
            "H100-SXM",
            "--output",
            "custom-report.md",
            "--bytes-per-value",
            "1",
            "--context-lengths",
            "1024",
            "2048",
            "--batch-sizes",
            "1",
            "8",
            "32",
            "--prompt-lengths",
            "256",
            "512",
            "--prefill-batch-size",
            "4",
        ]
    )

    assert arguments.model == "llama-3-8b"
    assert arguments.hardware == "H100-SXM"
    assert arguments.output == Path("custom-report.md")
    assert arguments.bytes_per_value == 1
    assert arguments.context_lengths == (1_024, 2_048)
    assert arguments.batch_sizes == (1, 8, 32)
    assert arguments.prompt_lengths == (256, 512)
    assert arguments.prefill_batch_size == 4


def test_parse_args_requires_model() -> None:
    """Reject command lines without a model."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--hardware",
                "L4",
            ]
        )


def test_parse_args_requires_hardware() -> None:
    """Reject command lines without hardware."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "qwen2.5-1.5b",
            ]
        )


def test_parse_args_rejects_non_integer_bytes_per_value() -> None:
    """Reject non-integer value-size arguments."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "qwen2.5-1.5b",
                "--hardware",
                "L4",
                "--bytes-per-value",
                "two",
            ]
        )
