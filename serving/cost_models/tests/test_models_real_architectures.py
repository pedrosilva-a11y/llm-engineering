"""Validation of parameter counting against published model architectures."""

import pytest

from serving.cost_models.catalog import LLAMA_3_8B, QWEN_2_5_1_5B
from serving.cost_models.models import ModelSpecification


@pytest.mark.parametrize(
    ("model", "low", "high"),
    [
        pytest.param(
            LLAMA_3_8B,
            7.8e9,
            8.2e9,
            id="llama-3-8b",
        ),
        pytest.param(
            QWEN_2_5_1_5B,
            1.50e9,
            1.58e9,
            id="qwen2.5-1.5b",
        ),
    ],
)
def test_total_params_within_advertised_range(
    model: ModelSpecification,
    low: float,
    high: float,
) -> None:
    """Guard against assumptions that only hold for toy configurations."""
    assert low < model.total_params < high


def test_llama_3_8b_kv_cache_per_token() -> None:
    """GQA with 8 KV heads gives 128 KiB per token in fp16."""
    assert LLAMA_3_8B.kv_bytes_per_token() == 128 * 1_024
