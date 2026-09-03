"""Generate Markdown reports for analytical LLM serving cost models."""

from collections.abc import Sequence

from serving.cost_models.cost_model import CostModel


def render_configuration(
    cost_model: CostModel,
) -> str:
    """Render the selected model, hardware, and precision configuration.

    Args:
        cost_model: Cost model containing the selected model and hardware.

    Returns:
        Configuration section rendered as Markdown.
    """
    model_weight_gib = cost_model.model_weight_bytes / 1_024**3
    kv_cache_kib_per_token = (
        cost_model.model.kv_bytes_per_token(
            bytes_per_value=cost_model.bytes_per_value,
        )
        / 1_024
    )

    lines = [
        "## Configuration",
        "",
        f"- Model: {cost_model.model.name}",
        f"- Hardware: {cost_model.hardware.name}",
        f"- Bytes per value: {cost_model.bytes_per_value}",
        f"- Model parameters: {cost_model.model.total_params:,}",
        f"- Model weight memory: {model_weight_gib:.2f} GiB",
        f"- KV cache per token: {kv_cache_kib_per_token:,.2f} KiB",
        (
            "- Hardware ridge point: "
            f"{cost_model.hardware.ridge_point_flops_per_byte:.2f} FLOPs/byte"
        ),
        "",
    ]

    return "\n".join(lines)


def render_memory_table(
    cost_model: CostModel,
    context_lengths: Sequence[int],
) -> str:
    """Render KV-cache memory and theoretical concurrency estimates.

    Args:
        cost_model: Cost model used to calculate memory estimates.
        context_lengths: Sequence lengths to evaluate.

    Returns:
        Memory-analysis section rendered as Markdown.
    """
    lines = [
        "## Memory Estimates",
        "",
        "| Context Length | KV cache / sequence | Max concurrency |",
        "| ---: | ---: | ---: |",
    ]

    for context_length in context_lengths:
        kv_cache_bytes = cost_model.kv_cache_bytes(
            sequence_length=context_length,
        )
        kv_cache_mib = kv_cache_bytes / 1_024**2
        max_concurrency = cost_model.max_concurrent_sequences(
            sequence_length=context_length,
        )

        lines.append(
            f"| {context_length:,} | {kv_cache_mib:,.2f} MiB | {max_concurrency:,} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_decode_intensity_table(
    cost_model: CostModel,
    context_lengths: Sequence[int],
    batch_sizes: Sequence[int],
) -> str:
    """Render decode arithmetic-intensity and roofline estimates.

    Args:
        cost_model: Cost model used to calculate decode estimates.
        context_lengths: Decode context lengths to evaluate.
        batch_sizes: Decode batch sizes to evaluate.

    Returns:
        Decode-analysis section rendered as Markdown.
    """
    ridge_point = cost_model.hardware.ridge_point_flops_per_byte

    lines = [
        "## Decode Estimates",
        "",
        (
            "| Context Length | Batch Size | FLOPs / token | "
            "Arithmetic Intensity | Ideal TPOT | Ideal Throughput | "
            "Ridge Point | Regime |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for context_length in context_lengths:
        flops_per_token = cost_model.estimated_decode_flops_per_token(
            context_length=context_length,
        )

        for batch_size in batch_sizes:
            intensity = cost_model.decode_arithmetic_intensity(
                context_length=context_length,
                batch_size=batch_size,
            )
            step_seconds = cost_model.estimated_decode_step_seconds(
                context_length=context_length,
                batch_size=batch_size,
            )
            tpot_ms = step_seconds * 1_000

            throughput = cost_model.estimated_decode_throughput_tokens_per_second(
                context_length=context_length,
                batch_size=batch_size,
            )
            regime = (
                "Compute-bound"
                if cost_model.is_compute_bound(intensity)
                else "Memory-bound"
            )

            lines.append(
                f"| {context_length:,} | "
                f"{batch_size:,} | "
                f"{flops_per_token:,.0f} | "
                f"{intensity:,.2f} FLOPs/byte | "
                f"{tpot_ms:,.2f} ms | "
                f"{throughput:,.2f} tokens/s | "
                f"{ridge_point:,.2f} FLOPs/byte | "
                f"{regime} |"
            )

    lines.append("")

    return "\n".join(lines)


def render_prefill_intensity_table(
    cost_model: CostModel,
    prompt_lengths: Sequence[int],
    batch_size: int = 1,
) -> str:
    """Render prefill arithmetic-intensity and roofline estimates.

    Args:
        cost_model: Cost model used to calculate prefill estimates.
        prompt_lengths: Prompt lengths to evaluate.
        batch_size: Number of prompts processed together.

    Returns:
        Prefill-analysis section rendered as Markdown.
    """
    ridge_point = cost_model.hardware.ridge_point_flops_per_byte

    lines = [
        "## Prefill Estimates",
        "",
        (
            "| Prompt Length | Batch Size | Arithmetic Intensity | "
            "Ideal Prefill Time | Ridge Point | Regime |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for prompt_length in prompt_lengths:
        intensity = cost_model.prefill_arithmetic_intensity(
            sequence_length=prompt_length,
            batch_size=batch_size,
        )
        prefill_seconds = cost_model.estimated_prefill_seconds(
            sequence_length=prompt_length,
            batch_size=batch_size,
        )
        prefill_ms = prefill_seconds * 1_000
        regime = (
            "Compute-bound"
            if cost_model.is_compute_bound(intensity)
            else "Memory-bound"
        )

        lines.append(
            f"| {prompt_length:,} | "
            f"{batch_size:,} | "
            f"{intensity:,.2f} FLOPs/byte | "
            f"{prefill_ms:,.2f} ms | "
            f"{ridge_point:,.2f} FLOPs/byte | "
            f"{regime} |"
        )

    lines.append("")

    return "\n".join(lines)


def render_report(
    cost_model: CostModel,
    context_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    prompt_lengths: Sequence[int],
    prefill_batch_size: int = 1,
) -> str:
    """Combine analytical cost-model sections into a Markdown report.

    Args:
        cost_model: Cost model used to generate the report.
        context_lengths: Context lengths used for memory and decode analysis.
        batch_sizes: Batch sizes used for decode analysis.
        prompt_lengths: Prompt lengths used for prefill analysis.
        prefill_batch_size: Batch size used for prefill analysis.

    Returns:
        Complete analytical report rendered as Markdown.
    """
    sections = [
        "# LLM Serving Cost-Model Predictions",
        "",
        render_configuration(cost_model),
        render_memory_table(
            cost_model=cost_model,
            context_lengths=context_lengths,
        ),
        render_decode_intensity_table(
            cost_model=cost_model,
            context_lengths=context_lengths,
            batch_sizes=batch_sizes,
        ),
        render_prefill_intensity_table(
            cost_model=cost_model,
            prompt_lengths=prompt_lengths,
            batch_size=prefill_batch_size,
        ),
    ]

    return "\n".join(sections)
