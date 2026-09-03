"""Generate an analytical LLM serving cost-model report."""

from serving.cost_models.catalog import HARDWARE_CATALOG, MODEL_CATALOG
from serving.cost_models.cli import parse_args
from serving.cost_models.cost_model import CostModel
from serving.cost_models.report import render_report


def main() -> None:
    """Generate and write the analytical cost-model report."""
    arguments = parse_args()

    model = MODEL_CATALOG[arguments.model]
    hardware = HARDWARE_CATALOG[arguments.hardware]

    cost_model = CostModel(
        model=model,
        hardware=hardware,
        bytes_per_value=arguments.bytes_per_value,
    )

    markdown = render_report(
        cost_model=cost_model,
        context_lengths=arguments.context_lengths,
        batch_sizes=arguments.batch_sizes,
        prompt_lengths=arguments.prompt_lengths,
        prefill_batch_size=arguments.prefill_batch_size,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        markdown,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
