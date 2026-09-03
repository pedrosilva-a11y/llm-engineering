"""Validation of cost-model estimates against published models and accelerators."""

import pytest

from serving.cost_models.catalog import H100_SXM, LLAMA_3_8B
from serving.cost_models.cost_model import CostModel
from serving.cost_models.models import ModelSpecification

# Same model with multi-head attention, used to isolate the effect of GQA.
LLAMA_3_8B_WITHOUT_GQA = ModelSpecification(
    name="llama-3-8b-mha",
    n_layer=32,
    d_model=4_096,
    n_head=32,
    n_kv_head=32,
    d_head=128,
    d_ff=14_336,
    vocab_size=128_256,
    norm_has_bias=False,
)


@pytest.fixture
def llama_on_h100() -> CostModel:
    """Create a cost model for Llama 3 8b served in bf16 on an H100."""
    return CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
    )


@pytest.fixture
def llama_mha_on_h100() -> CostModel:
    """Create a cost model for Llama 3 8b MHA in bf16 on an H100."""
    return CostModel(
        model=LLAMA_3_8B_WITHOUT_GQA,
        hardware=H100_SXM,
    )


def test_weight_memory_matches_expected_footprint(
    llama_on_h100: CostModel,
) -> None:
    """An 8B in bf16 occupies roughly 15 GiB of device memory."""
    weight_gib = llama_on_h100.model_weight_bytes / 1_024**3
    assert 14.5 < weight_gib < 15.5


def test_concurrency_ceiling_falls_with_context_length(
    llama_on_h100: CostModel,
) -> None:
    """Quadrupling context roughly quarters the number of resident sequences."""
    assert llama_on_h100.max_concurrent_sequences(2_048) == 258
    assert llama_on_h100.max_concurrent_sequences(8_192) == 64


def test_grouped_query_attention_multiplies_concurrency(
    llama_on_h100: CostModel,
    llama_mha_on_h100: CostModel,
) -> None:
    """Sharing 32 query heads across 8 KV heads quadruples resident sequences."""
    sequence_length = 2_048

    ratio = llama_on_h100.max_concurrent_sequences(
        sequence_length
    ) / llama_mha_on_h100.max_concurrent_sequences(sequence_length)

    assert ratio == pytest.approx(4.0, abs=0.2)


def test_single_stream_decode_is_deeply_memory_bound(
    llama_on_h100: CostModel,
) -> None:
    """Batch-1 decode reaches under 1% of the arithmetic intensity the GPU demands."""
    intensity = llama_on_h100.decode_arithmetic_intensity(
        context_length=2_048,
        batch_size=1,
    )

    assert intensity < 1.0
    assert not llama_on_h100.is_compute_bound(intensity)
    assert intensity / H100_SXM.ridge_point_flops_per_byte < 0.01


def test_decode_never_becomes_compute_bound_at_long_context(
    llama_on_h100: CostModel,
) -> None:
    """KV traffic caps decode intensity far below the ridge point at 2k context.

    At large batch sizes, intensity approaches the ratio between decode FLOPs
    per token and KV-cache bytes read per sequence for the current context.
    At 2k context this saturates near 56 FLOPs/byte, while the H100 ridge point
    is roughly 295 FLOPs/byte.
    """
    saturated = llama_on_h100.decode_arithmetic_intensity(
        context_length=2_048,
        batch_size=100_000,
    )

    assert saturated < 60
    assert not llama_on_h100.is_compute_bound(saturated)


def test_short_context_decode_can_reach_the_ridge_point(
    llama_on_h100: CostModel,
) -> None:
    """At 128-token context the KV term is small enough for batching to pay off."""
    intensity = llama_on_h100.decode_arithmetic_intensity(
        context_length=128,
        batch_size=2_048,
    )

    assert llama_on_h100.is_compute_bound(intensity)


def test_prefill_crosses_the_ridge_point_between_256_and_384_tokens(
    llama_on_h100: CostModel,
) -> None:
    """Prefill is compute-bound once the prompt is long enough to amortize weights."""
    short = llama_on_h100.prefill_arithmetic_intensity(
        sequence_length=256,
    )
    long = llama_on_h100.prefill_arithmetic_intensity(
        sequence_length=384,
    )

    assert not llama_on_h100.is_compute_bound(short)
    assert llama_on_h100.is_compute_bound(long)


def test_prefill_is_orders_of_magnitude_more_intense_than_decode(
    llama_on_h100: CostModel,
) -> None:
    """The two phases sit in different roofline regimes, which is why they differ."""
    prefill = llama_on_h100.prefill_arithmetic_intensity(
        sequence_length=2_048,
    )
    decode = llama_on_h100.decode_arithmetic_intensity(
        context_length=2_048,
        batch_size=1,
    )

    assert prefill / decode > 1_000


def test_one_byte_values_more_than_double_concurrency_ceiling() -> None:
    """One-byte weight and KV storage increases resident-sequence capacity."""
    bf16 = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
        bytes_per_value=2,
    )
    one_byte = CostModel(
        model=LLAMA_3_8B,
        hardware=H100_SXM,
        bytes_per_value=1,
    )

    assert one_byte.max_concurrent_sequences(2_048) > 2 * bf16.max_concurrent_sequences(
        2_048
    )
