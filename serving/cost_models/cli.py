"""Command-line interface for LLM serving cost-model reports."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CliArguments:
    """Parsed command-line arguments for cost-model report generation."""

    model: str
    hardware: str
    output: Path
    bytes_per_value: int
    context_lengths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    prefill_batch_size: int


def parse_args(
    argv: Sequence[str] | None = None,
) -> CliArguments:
    """Parse command-line arguments.

    Args:
        argv: Optional sequence of command-line arguments. Uses process
            arguments when omitted.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate analytical LLM serving cost-model predictions.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model specification name.",
    )
    parser.add_argument(
        "--hardware",
        required=True,
        help="Hardware specification name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("serving/cost_models/outputs/analytical_report.md"),
        help="Path for the generated Markdown report.",
    )
    parser.add_argument(
        "--bytes-per-value",
        type=int,
        default=2,
        help="Storage bytes used per model value.",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 1_024, 2_048, 4_096, 8_192],
        help="Context lengths used for memory and decode analysis.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        help="Batch sizes used for decode analysis.",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 384, 512, 1_024, 2_048],
        help="Prompt lengths used for prefill analysis.",
    )
    parser.add_argument(
        "--prefill-batch-size",
        type=int,
        default=1,
        help="Batch size used for prefill analysis.",
    )

    namespace = parser.parse_args(argv)

    return CliArguments(
        model=namespace.model,
        hardware=namespace.hardware,
        output=namespace.output,
        bytes_per_value=namespace.bytes_per_value,
        context_lengths=tuple(namespace.context_lengths),
        batch_sizes=tuple(namespace.batch_sizes),
        prompt_lengths=tuple(namespace.prompt_lengths),
        prefill_batch_size=namespace.prefill_batch_size,
    )
