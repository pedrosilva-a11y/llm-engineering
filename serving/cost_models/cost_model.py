"""Analytical cost model for LLM inference workloads."""

from dataclasses import dataclass

from serving.cost_models.hardware import HardwareSpecification
from serving.cost_models.models import ModelSpecification


@dataclass(frozen=True)
class CostModel:
    """Estimate memory and compute costs for serving an LLM.

    Combines a model architecture with accelerator specifications to estimate
    memory requirements, concurrency limits, arithmetic intensity, and roofline
    behavior.
    """

    model: ModelSpecification
    hardware: HardwareSpecification
    bytes_per_value: int = 2

    def __post_init__(self) -> None:
        """Validate cost-model configuration."""
        if self.bytes_per_value <= 0:
            raise ValueError("bytes_per_value must be greater than zero.")

    # Memory

    @property
    def model_weight_bytes(self) -> int:
        """Calculate memory occupied by model weights.

        Returns:
            Memory demanded by the model weights in bytes.
        """
        return self.model.weight_bytes(
            bytes_per_value=self.bytes_per_value,
        )

    @property
    def memory_available_for_kv_cache_bytes(self) -> float:
        """Calculate accelerator memory remaining after loading model weights.

        Returns:
            Available memory for KV cache in bytes.
        """
        return self.hardware.memory_capacity_bytes - self.model_weight_bytes

    def kv_cache_bytes(
        self,
        sequence_length: int,
        batch_size: int = 1,
    ) -> int:
        """Calculate total KV-cache memory for a batch of sequences.

        Args:
            sequence_length: Number of cached tokens per sequence.
            batch_size: Number of concurrent sequences.

        Returns:
            Total KV-cache memory in bytes.
        """
        if any(value <= 0 for value in (sequence_length, batch_size)):
            raise ValueError(
                "sequence_length and batch_size must be greater than zero.",
            )

        return (
            batch_size
            * sequence_length
            * self.model.kv_bytes_per_token(
                bytes_per_value=self.bytes_per_value,
            )
        )

    def max_concurrent_sequences(
        self,
        sequence_length: int,
    ) -> int:
        """Estimate the theoretical maximum number of concurrent sequences.

        Args:
            sequence_length: Number of cached tokens per sequence.

        Returns:
            Maximum number of sequences that fit in remaining accelerator
            memory.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if self.memory_available_for_kv_cache_bytes <= 0:
            raise ValueError("There is no memory available for the KV-cache.")

        max_concurrent_float = self.memory_available_for_kv_cache_bytes / (
            sequence_length
            * self.model.kv_bytes_per_token(
                bytes_per_value=self.bytes_per_value,
            )
        )
        return int(max_concurrent_float)

    # Compute

    def transformer_projection_flops_per_token(self) -> float:
        """Estimate transformer projection FLOPs required per token.

        Counts the dense attention and MLP projection work across all transformer
        layers. Each parameter interaction is approximated as two floating-point
        operations: one multiplication and one addition.

        Normalization and other elementwise operations are excluded.

        Returns:
            Estimated transformer projection FLOPs required per token.
        """
        projection_params_per_layer = (
            self.model.attention_params_per_layer + self.model.mlp_params_per_layer
        )

        return 2 * self.model.n_layer * projection_params_per_layer

    def attention_flops_per_token(
        self,
        context_length: int,
    ) -> float:
        """Estimate attention FLOPs required for one token.

        Counts the query-key dot products and the weighted-value aggregation
        across all attention heads and transformer layers. Softmax and other
        elementwise operations are excluded.

        Args:
            context_length: Number of tokens attended to by the current token.

        Returns:
            Estimated attention FLOPs.
        """
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero.")

        flops_per_layer = 4 * self.model.n_head * self.model.d_head * context_length

        return self.model.n_layer * flops_per_layer

    def estimated_decode_flops_per_token(
        self,
        context_length: int,
    ) -> float:
        """Estimate the dominant FLOP cost of decoding one token.

        Includes dense transformer projections and context-dependent attention
        operations. Normalization, activation functions, softmax, residual
        operations, and other lower-order costs are excluded.

        Args:
            context_length: Number of tokens attended to by the current token.

        Returns:
            Estimated dominant FLOPs required to generate one token.
        """
        return (
            self.transformer_projection_flops_per_token()
            + self.attention_flops_per_token(context_length)
        )

    def estimated_prefill_flops(
        self,
        sequence_length: int,
        batch_size: int = 1,
    ) -> float:
        """Estimate the dominant FLOP cost of prompt prefill.

        Includes transformer projection work for every prompt token and
        causal self-attention work across the prompt. Normalization,
        activation functions, softmax, residual operations, and other
        lower-order costs are excluded.

        Args:
            sequence_length: Number of prompt tokens processed.
            batch_size: Number of prompts processed together.

        Returns:
            Estimated dominant FLOPs required for prompt prefill.
        """
        if any(value <= 0 for value in (sequence_length, batch_size)):
            raise ValueError(
                "sequence_length and batch_size must be greater than zero.",
            )

        projection_flops = (
            batch_size * sequence_length * self.transformer_projection_flops_per_token()
        )

        attention_flops = (
            batch_size
            * 2
            * self.model.n_layer
            * self.model.n_head
            * self.model.d_head
            * sequence_length
            * (sequence_length + 1)
        )

        return projection_flops + attention_flops

    # Arithmetic intensity

    def decode_arithmetic_intensity(
        self,
        context_length: int,
        batch_size: int = 1,
    ) -> float:
        """Estimate arithmetic intensity during autoregressive decoding.

        Uses a first-order roofline approximation. The FLOP estimate includes
        the dominant transformer projection and context-dependent attention
        operations. Memory traffic is approximated as one effective read of the
        full model weights per decode step, shared across the batch, plus KV-cache
        reads for the existing context and KV-cache writes for newly generated
        tokens.

        Activation traffic, temporary buffers, cache effects, kernel behavior,
        and runtime overhead are excluded. Because the FLOP estimate omits some
        lower-order operations while weight traffic uses the full model size,
        this value should be interpreted as a coarse analytical estimate rather
        than an exact kernel-level measurement.

        When model-weight traffic dominates, arithmetic intensity grows
        approximately with batch size because the same weights are amortized
        across more sequences. At sufficiently large batch sizes or context
        lengths, KV-cache traffic can dominate, causing arithmetic intensity to
        grow sublinearly or approach a plateau.

        Args:
            context_length: Current sequence context length.
            batch_size: Number of sequences decoded together.

        Returns:
            Estimated arithmetic intensity in FLOPs per byte.
        """
        if any(value <= 0 for value in (context_length, batch_size)):
            raise ValueError(
                "context_length and batch_size must be greater than zero.",
            )

        total_flops = batch_size * self.estimated_decode_flops_per_token(context_length)

        kv_cache_read_bytes = self.kv_cache_bytes(
            sequence_length=context_length,
            batch_size=batch_size,
        )
        kv_cache_write_bytes = self.kv_cache_bytes(
            sequence_length=1,
            batch_size=batch_size,
        )

        total_bytes_moved = (
            self.model_weight_bytes + kv_cache_read_bytes + kv_cache_write_bytes
        )

        return total_flops / total_bytes_moved

    def prefill_arithmetic_intensity(
        self,
        sequence_length: int,
        batch_size: int = 1,
    ) -> float:
        """Estimate arithmetic intensity during prompt prefill.

        Uses a first-order roofline approximation. The FLOP estimate includes
        dominant transformer projection work and causal attention work. Memory
        traffic is approximated as one effective model-weight read plus KV-cache
        writes. Activation traffic, temporary buffers, cache effects, kernel behavior,
        and runtime overhead are excluded.

        Args:
            sequence_length: Number of prompt tokens processed.
            batch_size: Number of prompts processed together.

        Returns:
            Estimated arithmetic intensity in FLOPs per byte.
        """
        if any(value <= 0 for value in (sequence_length, batch_size)):
            raise ValueError(
                "sequence_length and batch_size must be greater than zero.",
            )

        total_flops = self.estimated_prefill_flops(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )

        kv_cache_write_bytes = self.kv_cache_bytes(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )

        total_bytes_moved = self.model_weight_bytes + kv_cache_write_bytes

        return total_flops / total_bytes_moved

    # Roofline

    def is_compute_bound(
        self,
        arithmetic_intensity: float,
    ) -> bool:
        """Determine whether a workload is compute-bound.

        Args:
            arithmetic_intensity: Workload arithmetic intensity in FLOPs
                per byte.

        Returns:
            True when arithmetic intensity exceeds the accelerator ridge point.
        """
        if arithmetic_intensity <= 0:
            raise ValueError("arithmetic_intensity must be greater than zero.")

        return arithmetic_intensity > self.hardware.ridge_point_flops_per_byte

    def achievable_flops_per_second(
        self,
        arithmetic_intensity: float,
    ) -> float:
        """Estimate ideal roofline-limited compute throughput.

        Args:
            arithmetic_intensity: Workload arithmetic intensity in FLOPs
                per byte.

        Returns:
            Ideal achievable throughput in FLOPs per second.
        """
        if arithmetic_intensity <= 0:
            raise ValueError(
                "arithmetic_intensity must be greater than zero.",
            )

        bandwidth_limited_flops = (
            self.hardware.memory_bandwidth_bytes_per_second * arithmetic_intensity
        )

        return min(
            self.hardware.peak_bf16_flops_per_second,
            bandwidth_limited_flops,
        )

    def estimated_prefill_seconds(
        self,
        sequence_length: int,
        batch_size: int = 1,
    ) -> float:
        """Estimate ideal roofline-limited prefill execution time.

        Args:
            sequence_length: Number of prompt tokens processed.
            batch_size: Number of prompts processed together.

        Returns:
            Ideal prefill execution time in seconds.
        """
        arithmetic_intensity = self.prefill_arithmetic_intensity(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        total_flops = self.estimated_prefill_flops(
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        achievable_flops = self.achievable_flops_per_second(
            arithmetic_intensity=arithmetic_intensity,
        )

        return total_flops / achievable_flops

    def estimated_decode_step_seconds(
        self,
        context_length: int,
        batch_size: int = 1,
    ) -> float:
        """Estimate ideal roofline-limited decode-step execution time.

        One decode step produces one new token for every sequence in the
        batch. The returned duration therefore also represents the idealized
        per-sequence time per output token for the step.

        Args:
            context_length: Current sequence context length.
            batch_size: Number of sequences decoded together.

        Returns:
            Ideal decode-step execution time in seconds.
        """
        if any(value <= 0 for value in (context_length, batch_size)):
            raise ValueError(
                "context_length and batch_size must be greater than zero.",
            )

        arithmetic_intensity = self.decode_arithmetic_intensity(
            context_length=context_length,
            batch_size=batch_size,
        )
        total_flops = batch_size * self.estimated_decode_flops_per_token(
            context_length=context_length,
        )
        achievable_flops = self.achievable_flops_per_second(
            arithmetic_intensity=arithmetic_intensity,
        )

        return total_flops / achievable_flops

    def estimated_decode_throughput_tokens_per_second(
        self,
        context_length: int,
        batch_size: int = 1,
    ) -> float:
        """Estimate ideal aggregate decode throughput.

        Args:
            context_length: Current sequence context length.
            batch_size: Number of sequences decoded together.

        Returns:
            Ideal aggregate decode throughput in tokens per second.
        """
        step_seconds = self.estimated_decode_step_seconds(
            context_length=context_length,
            batch_size=batch_size,
        )

        return batch_size / step_seconds
