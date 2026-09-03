"""Validation of published accelerator specifications."""

from serving.cost_models.catalog import H100_SXM, L4
from serving.cost_models.hardware import HardwareSpecification

A100_80GB_SXM = HardwareSpecification(
    name="A100-80GB-SXM",
    peak_bf16_tflops=312.0,
    memory_bandwidth_tb_s=2.039,
    memory_capacity_gib=79.6,
)


def test_ridge_points_are_in_expected_range() -> None:
    """Ridge points sanity-check the declared specifications."""
    assert 150 < A100_80GB_SXM.ridge_point_flops_per_byte < 160
    assert 290 < H100_SXM.ridge_point_flops_per_byte < 300
    assert 395 < L4.ridge_point_flops_per_byte < 410


def test_memory_wall_across_generations() -> None:
    """Compute grew far faster than bandwidth from A100 to H100.

    A strongly memory-bound workload therefore sees roughly 1.6x improvement,
    not the 3x the peak compute figures suggest.
    """
    compute_ratio = H100_SXM.peak_bf16_flops_per_second / (
        A100_80GB_SXM.peak_bf16_flops_per_second
    )
    bandwidth_ratio = H100_SXM.memory_bandwidth_bytes_per_second / (
        A100_80GB_SXM.memory_bandwidth_bytes_per_second
    )

    assert compute_ratio > 3.0
    assert bandwidth_ratio < 1.7
    assert H100_SXM.ridge_point_flops_per_byte > (
        A100_80GB_SXM.ridge_point_flops_per_byte
    )


def test_l4_demands_the_highest_arithmetic_intensity() -> None:
    """The rented card is the most bandwidth-starved of the three."""
    assert L4.ridge_point_flops_per_byte > H100_SXM.ridge_point_flops_per_byte
