"""Tests for analytical LLM inference cost modeling."""

import pytest

from serving.cost_models.cost_model import CostModel
from serving.cost_models.hardware import HardwareSpecification
from serving.cost_models.models import ModelSpecification


@pytest.fixture
def model() -> ModelSpecification:
    """Create a small decoder-only model with hand-checkable dimensions."""
    return ModelSpecification(
        name="test-model",
        n_layer=2,
        d_model=64,
        n_head=4,
        n_kv_head=2,
        d_head=16,
        d_ff=256,
        vocab_size=1_000,
        norm_has_bias=False,
    )


@pytest.fixture
def hardware() -> HardwareSpecification:
    """Create a synthetic accelerator with simple roofline characteristics."""
    return HardwareSpecification(
        name="test-accelerator",
        peak_bf16_tflops=100.0,
        memory_bandwidth_tb_s=2.0,
        memory_capacity_gib=1.0,
    )


@pytest.fixture
def cost_model(
    model: ModelSpecification,
    hardware: HardwareSpecification,
) -> CostModel:
    """Create a cost model using the synthetic model and accelerator."""
    return CostModel(
        model=model,
        hardware=hardware,
        bytes_per_value=2,
    )


# Configuration


@pytest.mark.parametrize(
    "bytes_per_value",
    [0, -1],
)
def test_non_positive_bytes_per_value_is_rejected(
    model: ModelSpecification,
    hardware: HardwareSpecification,
    bytes_per_value: int,
) -> None:
    """Reject cost-model configurations with invalid value sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        CostModel(
            model=model,
            hardware=hardware,
            bytes_per_value=bytes_per_value,
        )


# Memory Accounting


def test_model_weight_bytes(
    cost_model: CostModel,
) -> None:
    """Calculate model-weight memory using the configured value size."""
    assert cost_model.model_weight_bytes == 502_400


def test_memory_available_for_kv_cache(
    cost_model: CostModel,
) -> None:
    """Subtract model-weight memory from total accelerator capacity."""
    assert cost_model.memory_available_for_kv_cache_bytes == 1_073_239_424


def test_kv_cache_bytes(
    cost_model: CostModel,
) -> None:
    """Calculate KV-cache memory from sequence length and batch size."""
    batch_size = 4
    sequence_length = 128
    kv_cache_bytes = cost_model.kv_cache_bytes(
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    assert kv_cache_bytes == 131_072


@pytest.mark.parametrize(
    ("sequence_length", "batch_size"),
    [
        (-128, 4),
        (0, 4),
        (128, -4),
        (128, 0),
    ],
)
def test_non_positive_kv_cache_inputs_are_rejected(
    cost_model: CostModel,
    sequence_length: int,
    batch_size: int,
) -> None:
    """Reject non-positive sequence lengths and batch sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.kv_cache_bytes(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )


def test_kv_cache_scales_with_sequence_length(
    cost_model: CostModel,
) -> None:
    """Verify KV-cache memory grows linearly with sequence length."""
    sequence_length = 128
    batch_size = 4

    kv_cache_128 = cost_model.kv_cache_bytes(
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    kv_cache_256 = cost_model.kv_cache_bytes(
        sequence_length=256,
        batch_size=batch_size,
    )
    kv_cache_512 = cost_model.kv_cache_bytes(
        sequence_length=512,
        batch_size=batch_size,
    )

    assert kv_cache_256 == 2 * kv_cache_128
    assert kv_cache_512 == 4 * kv_cache_128


def test_kv_cache_scales_with_batch_size(
    cost_model: CostModel,
) -> None:
    """Verify KV-cache memory grows linearly with batch size."""
    sequence_length = 128
    batch_size = 4

    kv_cache_4 = cost_model.kv_cache_bytes(
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    kv_cache_8 = cost_model.kv_cache_bytes(
        sequence_length=sequence_length,
        batch_size=8,
    )
    kv_cache_16 = cost_model.kv_cache_bytes(
        sequence_length=sequence_length,
        batch_size=16,
    )

    assert kv_cache_8 == 2 * kv_cache_4
    assert kv_cache_16 == 4 * kv_cache_4


def test_max_concurrent_sequences(
    cost_model: CostModel,
) -> None:
    """Calculate the theoretical number of sequences that fit in memory."""
    sequence_length = 128
    max_concurrent_sequences = cost_model.max_concurrent_sequences(
        sequence_length=sequence_length,
    )

    assert max_concurrent_sequences == 32_752


@pytest.mark.parametrize(
    "sequence_length",
    [0, -1],
)
def test_non_positive_sequence_length_is_rejected(
    cost_model: CostModel,
    sequence_length: int,
) -> None:
    """Reject non-positive sequence lengths for concurrency estimates."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.max_concurrent_sequences(
            sequence_length=sequence_length,
        )


def test_max_concurrency_raises_when_model_does_not_fit(
    model: ModelSpecification,
) -> None:
    """Reject concurrency estimates when model weights exceed device memory."""
    hardware = HardwareSpecification(
        name="insufficient-memory",
        peak_bf16_tflops=100.0,
        memory_bandwidth_tb_s=2.0,
        memory_capacity_gib=4.6e-4,
    )
    with pytest.raises(ValueError, match="no memory available for the KV-cache"):
        CostModel(
            model=model,
            hardware=hardware,
        ).max_concurrent_sequences(sequence_length=128)


def test_max_concurrency_decreases_with_sequence_length(
    cost_model: CostModel,
) -> None:
    """Verify longer sequences reduce the theoretical concurrency ceiling."""
    concurrency_128 = cost_model.max_concurrent_sequences(
        sequence_length=128,
    )
    concurrency_256 = cost_model.max_concurrent_sequences(
        sequence_length=256,
    )
    concurrency_512 = cost_model.max_concurrent_sequences(
        sequence_length=512,
    )

    assert concurrency_128 > concurrency_256 > concurrency_512


# Compute


def test_transformer_projection_flops_per_token(
    cost_model: CostModel,
) -> None:
    """Calculate dense attention and MLP projection FLOPs per token."""
    assert cost_model.transformer_projection_flops_per_token() == 245_760


def test_attention_flops_per_token(
    cost_model: CostModel,
) -> None:
    """Calculate attention FLOPs for one token at a known context length."""
    context_length = 128
    assert (
        cost_model.attention_flops_per_token(
            context_length=context_length,
        )
        == 65_536
    )


@pytest.mark.parametrize(
    "context_length",
    [0, -1],
)
def test_non_positive_attention_context_is_rejected(
    cost_model: CostModel,
    context_length: int,
) -> None:
    """Reject non-positive context lengths for attention FLOP estimates."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.attention_flops_per_token(
            context_length=context_length,
        )


def test_attention_flops_scale_linearly_with_context_length(
    cost_model: CostModel,
) -> None:
    """Verify decode attention FLOPs grow linearly with context length."""
    flops_128 = cost_model.attention_flops_per_token(
        context_length=128,
    )
    flops_256 = cost_model.attention_flops_per_token(
        context_length=256,
    )
    flops_512 = cost_model.attention_flops_per_token(
        context_length=512,
    )

    assert flops_256 == 2 * flops_128
    assert flops_512 == 4 * flops_128


def test_estimated_decode_flops_per_token(
    cost_model: CostModel,
) -> None:
    """Combine projection and attention FLOPs into the decode estimate."""
    assert (
        cost_model.estimated_decode_flops_per_token(
            context_length=128,
        )
        == 311_296
    )


def test_estimated_decode_flops_increase_with_context_length(
    cost_model: CostModel,
) -> None:
    """Verify longer contexts increase the estimated decode FLOP cost."""
    decode_flops_128 = cost_model.estimated_decode_flops_per_token(
        context_length=128,
    )
    decode_flops_256 = cost_model.estimated_decode_flops_per_token(
        context_length=256,
    )
    decode_flops_512 = cost_model.estimated_decode_flops_per_token(
        context_length=512,
    )

    assert decode_flops_128 < decode_flops_256 < decode_flops_512


def test_estimated_prefill_flops(
    cost_model: CostModel,
) -> None:
    """Calculate prefill FLOPs for a known workload."""
    assert (
        cost_model.estimated_prefill_flops(
            sequence_length=128,
            batch_size=4,
        )
        == 142_737_408
    )


@pytest.mark.parametrize(
    ("sequence_length", "batch_size"),
    [
        (0, 4),
        (-1, 4),
        (128, 0),
        (128, -1),
    ],
)
def test_non_positive_prefill_flops_inputs_are_rejected(
    cost_model: CostModel,
    sequence_length: int,
    batch_size: int,
) -> None:
    """Reject non-positive prefill sequence lengths and batch sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.estimated_prefill_flops(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )


def test_estimated_prefill_flops_increase_with_sequence_length(
    cost_model: CostModel,
) -> None:
    """Verify longer prompts increase prefill FLOP cost."""
    flops_128 = cost_model.estimated_prefill_flops(
        sequence_length=128,
    )
    flops_256 = cost_model.estimated_prefill_flops(
        sequence_length=256,
    )
    flops_512 = cost_model.estimated_prefill_flops(
        sequence_length=512,
    )

    assert flops_128 < flops_256 < flops_512


# Arithmetic Intensity


def test_decode_arithmetic_intensity(
    cost_model: CostModel,
) -> None:
    """Calculate decode arithmetic intensity for a known workload."""
    context_length = 128
    batch_size = 4
    assert cost_model.decode_arithmetic_intensity(
        context_length=context_length,
        batch_size=batch_size,
    ) == pytest.approx(1.962477)


@pytest.mark.parametrize(
    ("context_length", "batch_size"),
    [
        (0, 4),
        (-1, 4),
        (128, 0),
        (128, -1),
    ],
)
def test_non_positive_decode_arithmetic_intensity_inputs_are_rejected(
    cost_model: CostModel,
    context_length: int,
    batch_size: int,
) -> None:
    """Reject non-positive decode context lengths and batch sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.decode_arithmetic_intensity(
            context_length=context_length,
            batch_size=batch_size,
        )


def test_decode_arithmetic_intensity_increases_with_batch_size(
    cost_model: CostModel,
) -> None:
    """Verify batching increases decode arithmetic intensity."""
    context_length = 128

    decode_intensity_4 = cost_model.decode_arithmetic_intensity(
        context_length=context_length,
        batch_size=4,
    )
    decode_intensity_8 = cost_model.decode_arithmetic_intensity(
        context_length=context_length,
        batch_size=8,
    )
    decode_intensity_16 = cost_model.decode_arithmetic_intensity(
        context_length=context_length,
        batch_size=16,
    )

    assert decode_intensity_4 < decode_intensity_8 < decode_intensity_16


def test_decode_intensity_growth_rate_decays_at_large_batches(
    cost_model: CostModel,
) -> None:
    """Marginal intensity per added sequence shrinks as KV traffic grows."""
    ai = cost_model.decode_arithmetic_intensity
    early_rate = (ai(128, 8) - ai(128, 4)) / 4
    late_rate = (ai(128, 512) - ai(128, 256)) / 256
    assert early_rate > late_rate


def test_prefill_arithmetic_intensity(
    cost_model: CostModel,
) -> None:
    """Calculate prefill arithmetic intensity for a known workload."""
    sequence_length = 128
    batch_size = 4

    assert cost_model.prefill_arithmetic_intensity(
        sequence_length=sequence_length,
        batch_size=batch_size,
    ) == pytest.approx(225.3255)


@pytest.mark.parametrize(
    ("sequence_length", "batch_size"),
    [
        (0, 4),
        (-1, 4),
        (128, 0),
        (128, -1),
    ],
)
def test_non_positive_prefill_arithmetic_intensity_inputs_are_rejected(
    cost_model: CostModel,
    sequence_length: int,
    batch_size: int,
) -> None:
    """Reject non-positive prefill sequence lengths and batch sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.prefill_arithmetic_intensity(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )


def test_prefill_arithmetic_intensity_increases_with_sequence_length(
    cost_model: CostModel,
) -> None:
    """Verify longer prompts increase prefill arithmetic intensity."""
    batch_size = 4

    intensity_128 = cost_model.prefill_arithmetic_intensity(
        sequence_length=128,
        batch_size=batch_size,
    )
    intensity_256 = cost_model.prefill_arithmetic_intensity(
        sequence_length=256,
        batch_size=batch_size,
    )
    intensity_512 = cost_model.prefill_arithmetic_intensity(
        sequence_length=512,
        batch_size=batch_size,
    )

    assert intensity_128 < intensity_256 < intensity_512


def test_prefill_arithmetic_intensity_increases_with_batch_size(
    cost_model: CostModel,
) -> None:
    """Verify batching increases prefill arithmetic intensity."""
    sequence_length = 128

    intensity_4 = cost_model.prefill_arithmetic_intensity(
        sequence_length=sequence_length,
        batch_size=4,
    )
    intensity_8 = cost_model.prefill_arithmetic_intensity(
        sequence_length=sequence_length,
        batch_size=8,
    )
    intensity_16 = cost_model.prefill_arithmetic_intensity(
        sequence_length=sequence_length,
        batch_size=16,
    )

    assert intensity_4 < intensity_8 < intensity_16


def test_prefill_has_higher_intensity_than_single_batch_decode(
    cost_model: CostModel,
) -> None:
    """Verify representative prefill work has higher intensity than decode."""
    sequence_length = 128
    batch_size = 1

    decode_intensity = cost_model.decode_arithmetic_intensity(
        context_length=sequence_length,
        batch_size=batch_size,
    )
    prefill_intensity = cost_model.prefill_arithmetic_intensity(
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    assert prefill_intensity > decode_intensity


# Roofline


def test_arithmetic_intensity_below_ridge_point_is_not_compute_bound(
    cost_model: CostModel,
) -> None:
    """Classify workloads below the ridge point as memory-bound."""
    arithmetic_intensity = 49.98

    assert (
        cost_model.is_compute_bound(
            arithmetic_intensity=arithmetic_intensity,
        )
        is False
    )


def test_arithmetic_intensity_above_ridge_point_is_compute_bound(
    cost_model: CostModel,
) -> None:
    """Classify workloads above the ridge point as compute-bound."""
    arithmetic_intensity = 50.01

    assert (
        cost_model.is_compute_bound(
            arithmetic_intensity=arithmetic_intensity,
        )
        is True
    )


def test_arithmetic_intensity_at_ridge_point_is_not_strictly_compute_bound(
    cost_model: CostModel,
) -> None:
    """Treat the exact ridge point as not strictly compute-bound."""
    arithmetic_intensity = 50.0

    assert (
        cost_model.is_compute_bound(
            arithmetic_intensity=arithmetic_intensity,
        )
        is False
    )


@pytest.mark.parametrize(
    "arithmetic_intensity",
    [0.0, -1.0],
)
def test_non_positive_arithmetic_intensity_is_rejected(
    cost_model: CostModel,
    arithmetic_intensity: float,
) -> None:
    """Reject non-positive arithmetic-intensity values."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.is_compute_bound(
            arithmetic_intensity=arithmetic_intensity,
        )


@pytest.mark.parametrize(
    ("arithmetic_intensity", "expected_flops"),
    [
        (25.0, 50e12),
        (50.0, 100e12),
        (75.0, 100e12),
    ],
)
def test_achievable_flops_per_second(
    cost_model: CostModel,
    arithmetic_intensity: float,
    expected_flops: float,
) -> None:
    """Apply the roofline minimum of bandwidth and peak compute."""
    assert (
        cost_model.achievable_flops_per_second(
            arithmetic_intensity=arithmetic_intensity,
        )
        == expected_flops
    )


@pytest.mark.parametrize(
    "arithmetic_intensity",
    [0.0, -1.0],
)
def test_non_positive_achievable_flops_intensity_is_rejected(
    cost_model: CostModel,
    arithmetic_intensity: float,
) -> None:
    """Reject non-positive intensity for roofline throughput estimates."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.achievable_flops_per_second(
            arithmetic_intensity=arithmetic_intensity,
        )


def test_estimated_prefill_seconds(
    cost_model: CostModel,
) -> None:
    """Calculate ideal roofline prefill time for a known workload."""
    assert cost_model.estimated_prefill_seconds(
        sequence_length=128,
        batch_size=4,
    ) == pytest.approx(1.42737408e-6)


def test_estimated_decode_step_seconds(
    cost_model: CostModel,
) -> None:
    """Calculate ideal roofline decode-step time for a known workload."""
    assert cost_model.estimated_decode_step_seconds(
        context_length=128,
        batch_size=4,
    ) == pytest.approx(3.17248e-7)


@pytest.mark.parametrize(
    ("context_length", "batch_size"),
    [
        (0, 4),
        (-1, 4),
        (128, 0),
        (128, -1),
    ],
)
def test_non_positive_decode_step_inputs_are_rejected(
    cost_model: CostModel,
    context_length: int,
    batch_size: int,
) -> None:
    """Reject non-positive decode context lengths and batch sizes."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        cost_model.estimated_decode_step_seconds(
            context_length=context_length,
            batch_size=batch_size,
        )


def test_decode_step_time_increases_with_batch_size(
    cost_model: CostModel,
) -> None:
    """Verify larger decode batches take longer per decode step."""
    time_4 = cost_model.estimated_decode_step_seconds(
        context_length=128,
        batch_size=4,
    )
    time_16 = cost_model.estimated_decode_step_seconds(
        context_length=128,
        batch_size=16,
    )
    time_32 = cost_model.estimated_decode_step_seconds(
        context_length=128,
        batch_size=32,
    )

    assert time_4 < time_16 < time_32


def test_decode_throughput_increases_with_batch_size(
    cost_model: CostModel,
) -> None:
    """Verify batching improves aggregate decode throughput."""
    throughput_4 = cost_model.estimated_decode_throughput_tokens_per_second(
        context_length=128,
        batch_size=4,
    )
    throughput_16 = cost_model.estimated_decode_throughput_tokens_per_second(
        context_length=128,
        batch_size=16,
    )
    throughput_32 = cost_model.estimated_decode_throughput_tokens_per_second(
        context_length=128,
        batch_size=32,
    )

    assert throughput_4 < throughput_16 < throughput_32


def test_decode_throughput_growth_rate_decays_at_large_batches(
    cost_model: CostModel,
) -> None:
    """Verify marginal throughput gains shrink at large decode batches."""
    throughput = cost_model.estimated_decode_throughput_tokens_per_second

    early_rate = (
        throughput(context_length=128, batch_size=8)
        - throughput(context_length=128, batch_size=4)
    ) / 4

    late_rate = (
        throughput(context_length=128, batch_size=512)
        - throughput(context_length=128, batch_size=256)
    ) / 256

    assert early_rate > late_rate
