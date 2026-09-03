"""Test for hardware specifications."""

import pytest

from serving.cost_models.hardware import HardwareSpecification


@pytest.fixture
def accelerator() -> HardwareSpecification:
    """Create a synthetic accelerator with round numbers."""
    return HardwareSpecification(
        name="test-accelerator",
        peak_bf16_tflops=100.0,
        memory_bandwidth_tb_s=2.0,
        memory_capacity_gib=1.0,
    )


def test_unit_conversions(
    accelerator: HardwareSpecification,
) -> None:
    """Verify each declared unit converts to its base unit."""
    assert accelerator.peak_bf16_flops_per_second == 1e14
    assert accelerator.memory_bandwidth_bytes_per_second == 2e12
    assert accelerator.memory_capacity_bytes == 1_073_741_824


def test_memory_capacity_uses_binary_units(
    accelerator: HardwareSpecification,
) -> None:
    """GiB is 1024**3 bytes, not the vendor's decimal GB figure."""
    assert accelerator.memory_capacity_bytes != 1e9
    assert accelerator.memory_capacity_bytes == 1_073_741_824


def test_ridge_point(
    accelerator: HardwareSpecification,
) -> None:
    """Ridge point is peak compute divided by peak bandwidth."""
    assert accelerator.ridge_point_flops_per_byte == 50.0


@pytest.mark.parametrize(
    ("peak_bf16_tflops", "memory_bandwidth_tb_s", "memory_capacity_gib"),
    [
        (0.0, 2.0, 1.0),
        (-1.0, 2.0, 1.0),
        (100.0, 0.0, 1.0),
        (100.0, -1.0, 1.0),
        (100.0, 2.0, 0.0),
        (100.0, 2.0, -1.0),
    ],
)
def test_non_positive_values_are_rejected(
    peak_bf16_tflops: float,
    memory_bandwidth_tb_s: float,
    memory_capacity_gib: float,
) -> None:
    """Verify every numeric field rejects zero and negative values."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        HardwareSpecification(
            name="invalid",
            peak_bf16_tflops=peak_bf16_tflops,
            memory_bandwidth_tb_s=memory_bandwidth_tb_s,
            memory_capacity_gib=memory_capacity_gib,
        )
