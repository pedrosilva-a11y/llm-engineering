"""Data structure for hardware specifications that will host the LLM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpecification:
    """Specifications for an accelerator used for Large Language Model inference.

    Attributes:
        name: Human-readable accelerator identifier, such as ``L4`` or
            ``A100-80GB``.
        peak_bf16_tflops: Peak dense BF16 compute throughput, expressed in tera
            floating-point operations per second (TFLOP/s). Use the dense value
            rather than sparsity-accelerated marketing figures.
        memory_bandwidth_tb_s: Peak accelerator-memory bandwidth, expressed in terabytes
            per second (TB/s). This represents the theoretical maximum rate at which
            model weights and activations can be read from or written to device memory.
        memory_capacity_gib: Usable accelerator memory capacity, expressed in gibibytes
            (GiB), where one GiB equals 1024**3 bytes. Prefer the device-reported usable
            capacity rather than the vendor-advertised decimal GB figure. This value
            represents memory available before accounting for model weights, KV cache,
            runtime allocation, and framework overhead.
    """

    name: str
    peak_bf16_tflops: float
    memory_bandwidth_tb_s: float
    memory_capacity_gib: float

    def __post_init__(self) -> None:
        """Validate that hardware specification values are positive.

        Raises:
            ValueError: If any numeric hardware specification is not positive.
        """
        if any(
            value <= 0
            for value in (
                self.peak_bf16_tflops,
                self.memory_bandwidth_tb_s,
                self.memory_capacity_gib,
            )
        ):
            raise ValueError("Hardware specification values must be greater than zero.")

    @property
    def peak_bf16_flops_per_second(self) -> float:
        """Convert peak BF16 throughput from TFLOP/s to FLOP/s.

        Returns:
            Peak dense BF16 compute throughput in floating-point operations
            per second.
        """
        return self.peak_bf16_tflops * 1e12

    @property
    def memory_bandwidth_bytes_per_second(self) -> float:
        """Convert peak memory bandwidth from TB/s to bytes/s.

        Returns:
            Peak memory bandwidth in bytes per second.
        """
        return self.memory_bandwidth_tb_s * 1e12

    @property
    def memory_capacity_bytes(self) -> float:
        """Convert usable memory capacity from GiB to bytes.

        Returns:
            Total hardware memory capacity in bytes.
        """
        return self.memory_capacity_gib * 1_024**3

    @property
    def ridge_point_flops_per_byte(self) -> float:
        """Calculate the hardware roofline ridge point.

        The ridge point represents the arithmetic intensity required for a
        workload to transition from being memory-bandwidth-bound to
        compute-bound on this accelerator. Workloads below this threshold
        are primarily limited by memory bandwidth, while workloads above it
        are primarily limited by peak compute throughput.

        Returns:
            Hardware ridge point in floating-point operations per byte.
        """
        return self.peak_bf16_flops_per_second / self.memory_bandwidth_bytes_per_second
