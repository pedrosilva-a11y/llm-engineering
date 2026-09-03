"""Tests for Markdown cost-model report rendering."""

from serving.cost_models.catalog import H100_SXM, LLAMA_3_8B
from serving.cost_models.cost_model import CostModel
from serving.cost_models.report import (
    render_configuration,
    render_decode_intensity_table,
    render_memory_table,
    render_prefill_intensity_table,
    render_report,
)


def test_render_configuration() -> None:
    """Render model and hardware configuration as Markdown."""
    cost_model = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
    )

    rendered = render_configuration(cost_model=cost_model)

    assert "## Configuration" in rendered
    assert "- Model: llama-3-8b" in rendered
    assert "- Hardware: H100-SXM" in rendered
    assert "- Bytes per value: 2" in rendered
    assert "- KV cache per token: 128.00 KiB" in rendered
    assert "FLOPs/byte" in rendered


def test_render_memory_table() -> None:
    """Render KV-cache and concurrency estimates as a Markdown table."""
    cost_model = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
    )

    rendered = render_memory_table(
        cost_model=cost_model,
        context_lengths=(2_048, 8_192),
    )

    assert "## Memory Estimates" in rendered
    assert "| 2,048 | 256.00 MiB | 258 |" in rendered
    assert "| 8,192 | 1,024.00 MiB | 64 |" in rendered


def test_render_decode_intensity_table() -> None:
    """Render decode roofline estimates as Markdown table."""
    cost_model = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
    )

    rendered = render_decode_intensity_table(
        cost_model=cost_model,
        context_lengths=(2_048,),
        batch_sizes=(1, 32),
    )

    assert "## Decode Estimates" in rendered
    assert "Ideal TPOT" in rendered
    assert "Ideal Throughput" in rendered

    assert (
        "| 2,048 | 1 | 15,032,385,536 | "
        "0.92 FLOPs/byte | 4.87 ms | 205.16 tokens/s | "
        "295.37 FLOPs/byte | Memory-bound |"
    ) in rendered

    assert (
        "| 2,048 | 32 | 15,032,385,536 | "
        "19.51 FLOPs/byte | 7.36 ms | 4,348.06 tokens/s | "
        "295.37 FLOPs/byte | Memory-bound |"
    ) in rendered


def test_render_prefill_intensity_table() -> None:
    """Render prefill roofline estimates as a Markdown table."""
    cost_model = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
    )

    rendered = render_prefill_intensity_table(
        cost_model=cost_model,
        prompt_lengths=(256, 384),
    )

    assert "## Prefill Estimates" in rendered
    assert "Ideal Prefill Time" in rendered

    assert (
        "| 256 | 1 | 223.10 FLOPs/byte | 4.80 ms | 295.37 FLOPs/byte | Memory-bound |"
    ) in rendered

    assert (
        "| 384 | 1 | 335.11 FLOPs/byte | 5.46 ms | 295.37 FLOPs/byte | Compute-bound |"
    ) in rendered


def test_render_report() -> None:
    """Combine all analytical sections into a Markdown report."""
    cost_model = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
    )

    rendered = render_report(
        cost_model=cost_model,
        context_lengths=(2_048, 8_192),
        batch_sizes=(1, 32),
        prompt_lengths=(256, 384),
    )

    section_titles = [
        "# LLM Serving Cost-Model Predictions",
        "## Configuration",
        "## Memory Estimates",
        "## Decode Estimates",
        "## Prefill Estimates",
    ]
    for title in section_titles:
        assert title in rendered
