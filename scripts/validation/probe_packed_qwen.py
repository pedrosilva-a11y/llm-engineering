"""Probe Qwen compatibility with packed model execution."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache

from serving.engine.attention_mask import build_attention_mask
from serving.engine.batch_layout import BatchLayout
from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.packed_inputs import build_packed_inputs
from serving.engine.request import Request
from serving.engine.sequence import SequenceState

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

LayerKV = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ProbeComparison:
    """Summarize independent and packed next-token predictions.

    Attributes:
        independent_top1: Top-1 token from each independent execution.
        packed_top1: Top-1 token from each execution in the packed forward.
        max_abs_logit_diffs: Maximum absolute logit difference per execution.
    """

    independent_top1: tuple[int, ...]
    packed_top1: tuple[int, ...]
    max_abs_logit_diffs: tuple[float, ...]

    @property
    def all_top1_match(self) -> bool:
        """Return whether every packed top-1 matches its independent reference."""
        return self.independent_top1 == self.packed_top1


@dataclass(frozen=True)
class MixedProbeResult:
    """Summarize mixed decode-plus-prefill probe behavior.

    Attributes:
        comparison: Independent-versus-packed next-token comparison.
        model_mask_shape: Shape of the explicit model-facing attention mask.
        received_mask_shape: Shape observed by Qwen's first attention layer.
        mask_preserved: Whether the first attention layer received the explicit
            mask unchanged.
    """

    comparison: ProbeComparison
    model_mask_shape: tuple[int, ...]
    received_mask_shape: tuple[int, ...]
    mask_preserved: bool


class ProbeMixedCache(Cache):
    """Assemble packed mixed-phase K/V states for compatibility probing.

    This cache is intentionally probe-specific. It models exactly one decode
    execution followed by one prefill execution.

    The decode execution already owns a cached prefix. Qwen supplies new K/V for
    the packed query region containing one decode token followed by all prefill
    tokens. For each transformer layer this adapter returns K/V ordered as:

    ``decode prefix + decode newest token + prefill history``

    That ordering matches the packed key geometry represented by ``BatchLayout``.
    """

    def __init__(
        self,
        prefix_layers: tuple[LayerKV, ...],
        *,
        decode_query_length: int,
        prefill_query_length: int,
        expected_key_tokens: int,
    ) -> None:
        """Initialize the probe cache.

        Args:
            prefix_layers: Frozen per-layer K/V tensors for the decode request's
                already-cached prefix.
            decode_query_length: Number of new query tokens contributed by the
                decode execution.
            prefill_query_length: Number of query tokens contributed by prefill.
            expected_key_tokens: Total packed key length expected after combining
                cached and newly produced states.
        """
        super().__init__(layers=[])

        self._prefix_layers = prefix_layers
        self._decode_query_length = decode_query_length
        self._prefill_query_length = prefill_query_length
        self._expected_key_tokens = expected_key_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine cached decode history with newly produced packed K/V.

        Args:
            key_states: New packed key states for the current transformer layer.
            value_states: New packed value states for the current layer.
            layer_idx: Transformer layer whose states are being updated.
            *args: Additional Hugging Face cache arguments, unused by the probe.
            **kwargs: Additional Hugging Face cache arguments, unused by the probe.

        Returns:
            Packed full-history key and value tensors for attention.

        Raises:
            RuntimeError: If Qwen supplies unexpected packed K/V geometry.
        """
        del args, kwargs

        expected_query_tokens = self._decode_query_length + self._prefill_query_length

        if key_states.shape[-2] != expected_query_tokens:
            raise RuntimeError(
                "Unexpected packed key query length: "
                f"{key_states.shape[-2]} != {expected_query_tokens}."
            )

        if value_states.shape[-2] != expected_query_tokens:
            raise RuntimeError(
                "Unexpected packed value query length: "
                f"{value_states.shape[-2]} != {expected_query_tokens}."
            )

        if layer_idx >= len(self._prefix_layers):
            raise RuntimeError(
                f"Missing cached prefix states for transformer layer {layer_idx}."
            )

        prefix_keys, prefix_values = self._prefix_layers[layer_idx]

        decode_end = self._decode_query_length

        decode_keys = key_states[..., :decode_end, :]
        decode_values = value_states[..., :decode_end, :]

        prefill_keys = key_states[..., decode_end:, :]
        prefill_values = value_states[..., decode_end:, :]

        full_keys = torch.cat(
            (
                prefix_keys,
                decode_keys,
                prefill_keys,
            ),
            dim=-2,
        )

        full_values = torch.cat(
            (
                prefix_values,
                decode_values,
                prefill_values,
            ),
            dim=-2,
        )

        if full_keys.shape[-2] != self._expected_key_tokens:
            raise RuntimeError(
                "Probe cache produced unexpected key length: "
                f"{full_keys.shape[-2]} != {self._expected_key_tokens}."
            )

        if full_values.shape[-2] != self._expected_key_tokens:
            raise RuntimeError(
                "Probe cache produced unexpected value length: "
                f"{full_values.shape[-2]} != {self._expected_key_tokens}."
            )

        return full_keys, full_values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Fail if Qwen requests a scalar cache length for the packed probe.

        Raises:
            RuntimeError: Always. Mixed packed sequences have independent
                history lengths and therefore no meaningful single sequence
                length.
        """
        raise RuntimeError(
            "Unexpected get_seq_length() call in mixed packed probe "
            f"for layer {layer_idx}."
        )

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Fail if Qwen requests one scalar packed-query offset.

        Raises:
            RuntimeError: Always. Query offsets are sequence-local and supplied
                explicitly through ``position_ids``.
        """
        raise RuntimeError(
            "Unexpected get_query_offset() call in mixed packed probe "
            f"for layer {layer_idx}."
        )

    def get_mask_sizes(
        self,
        query_length: int,
        layer_idx: int,
    ) -> tuple[int, int]:
        """Fail if Qwen attempts to derive mask geometry from the cache.

        Args:
            query_length: Query length requested by Transformers.
            layer_idx: Transformer layer whose mask geometry was requested.

        Raises:
            RuntimeError: Always. The probe supplies a complete explicit 4-D
                attention mask and should not need cache-derived mask geometry.
        """
        raise RuntimeError(
            "Unexpected get_mask_sizes() call in mixed packed probe: "
            f"query_length={query_length}, layer_idx={layer_idx}."
        )


def load_qwen_for_probe(device: torch.device) -> PreTrainedModel:
    """Load the frozen Qwen checkpoint using eager attention.

    The CPU probe uses float32 because its purpose is validating execution
    semantics rather than reproducing the final CUDA precision configuration.

    Args:
        device: Device on which the model should execute.

    Returns:
        Qwen causal language model in evaluation mode.
    """
    model_factory: Any = AutoModelForCausalLM

    loaded_model: Any = model_factory.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    loaded_model = loaded_model.to(device)
    model = cast(PreTrainedModel, loaded_model)

    cast(torch.nn.Module, model).eval()

    return model


def make_prefill_execution(
    request_id: str,
    token_ids: tuple[int, ...],
    *,
    slot_start: int,
) -> ModelExecution:
    """Build one synthetic prefill execution for the compatibility probe.

    Args:
        request_id: Identifier for the synthetic request.
        token_ids: Complete current token history for the request.
        slot_start: First synthetic physical KV slot assigned to the sequence.

    Returns:
        Prefill model execution covering the complete token history.
    """
    request = Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        max_new_tokens=1,
    )

    sequence = SequenceState(request=request)

    slot_ids = tuple(
        range(
            slot_start,
            slot_start + sequence.current_length,
        )
    )

    return ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=slot_ids,
    )


def make_decode_execution(
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: tuple[int, ...],
    *,
    slot_start: int,
) -> ModelExecution:
    """Build one synthetic decode execution for the compatibility probe.

    Args:
        request_id: Identifier for the synthetic request.
        prompt_token_ids: Original prompt token identifiers.
        generated_token_ids: Tokens already appended to the sequence. The final
            token becomes the current decode query.
        slot_start: First synthetic physical KV slot assigned to the sequence.

    Returns:
        Decode model execution covering the complete current history.

    Raises:
        ValueError: If no generated token is supplied for decode.
    """
    if not generated_token_ids:
        raise ValueError("Decode probe requires at least one generated token.")

    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=16,
    )

    sequence = SequenceState(
        request=request,
        generated_token_ids=list(generated_token_ids),
    )

    slot_ids = tuple(
        range(
            slot_start,
            slot_start + sequence.current_length,
        )
    )

    return ModelExecution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=slot_ids,
    )


def semantic_to_additive_attention_mask(
    semantic_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert the semantic boolean mask into a 4-D additive attention mask.

    Allowed positions become zero and blocked positions become the minimum
    representable floating-point value. Dimensions are expanded to
    ``(1, 1, query_tokens, key_tokens)`` for the Qwen eager-attention probe.

    Args:
        semantic_mask: Two-dimensional boolean query-by-key attention mask.
        dtype: Floating-point dtype for the model-facing additive mask.

    Returns:
        Four-dimensional additive attention mask.

    Raises:
        ValueError: If the semantic mask is not two-dimensional and boolean, or
            if the requested output dtype is not floating point.
    """
    if semantic_mask.ndim != 2:
        raise ValueError("Semantic attention mask must be two-dimensional.")

    if semantic_mask.dtype != torch.bool:
        raise ValueError("Semantic attention mask must use boolean dtype.")

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Additive attention mask must use a floating-point dtype.")

    additive_mask = torch.zeros(
        semantic_mask.shape,
        dtype=dtype,
        device=semantic_mask.device,
    )

    additive_mask.masked_fill_(
        ~semantic_mask,
        torch.finfo(dtype).min,
    )

    return additive_mask.unsqueeze(0).unsqueeze(0)


def run_independent_prefill_logits(
    model: PreTrainedModel,
    execution: ModelExecution,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Run one prefill independently and return its next-token logits.

    Args:
        model: Causal language model under test.
        execution: Prefill execution to evaluate independently.
        device: Device on which model inputs should be created.

    Returns:
        Next-token logits with shape ``(vocab_size,)``.

    Raises:
        ValueError: If the execution is not a prefill execution.
    """
    if execution.phase is not ExecutionPhase.PREFILL:
        raise ValueError("Independent prefill probe requires PREFILL execution.")

    input_ids = torch.tensor(
        execution.sequence.all_token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    position_ids = torch.arange(
        execution.sequence.current_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    model_callable: Any = model

    with torch.inference_mode():
        outputs: Any = model_callable(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )

    logits = cast(torch.Tensor, outputs.logits)

    return logits[0, -1, :]


def build_decode_prefix_cache(
    model: PreTrainedModel,
    prefix_token_ids: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[DynamicCache, tuple[LayerKV, ...]]:
    """Build a normal HF cache for a decode request's existing prefix.

    A frozen copy of each layer's K/V is taken before the independent decode
    reference mutates the DynamicCache. Those copies seed ``ProbeMixedCache``.

    Args:
        model: Causal language model under test.
        prefix_token_ids: Tokens whose K/V should already exist before decode.
        device: Device on which cache construction should run.

    Returns:
        Mutable DynamicCache for the independent reference and immutable
        snapshots of the prefix K/V for every transformer layer.

    Raises:
        RuntimeError: If Hugging Face does not initialize K/V for every layer.
    """
    cache = DynamicCache(config=model.config)

    input_ids = torch.tensor(
        prefix_token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    position_ids = torch.arange(
        len(prefix_token_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    model_callable: Any = model

    with torch.inference_mode():
        model_callable(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    prefix_layers: list[LayerKV] = []

    for layer_idx, layer in enumerate(cache.layers):
        if not isinstance(layer, CacheLayerMixin):
            raise RuntimeError(
                "Packed Qwen probe expected a standard attention cache layer, "
                f"got {type(layer).__name__} at layer {layer_idx}."
            )

        keys = layer.keys
        values = layer.values

        if keys is None or values is None:
            raise RuntimeError(f"Prefix cache layer {layer_idx} was not initialized.")

        prefix_layers.append(
            (
                keys.detach().clone(),
                values.detach().clone(),
            )
        )

    return cache, tuple(prefix_layers)


def run_independent_decode_logits(
    model: PreTrainedModel,
    cache: DynamicCache,
    *,
    token_id: int,
    logical_position: int,
    device: torch.device,
) -> torch.Tensor:
    """Run one decode token against a normal Hugging Face cache.

    Args:
        model: Causal language model under test.
        cache: DynamicCache containing the token's previous logical history.
        token_id: Current decode query token.
        logical_position: Sequence-local position of the decode token.
        device: Device on which model inputs should be created.

    Returns:
        Next-token logits with shape ``(vocab_size,)``.
    """
    input_ids = torch.tensor(
        [[token_id]],
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.tensor(
        [[logical_position]],
        dtype=torch.long,
        device=device,
    )

    model_callable: Any = model

    with torch.inference_mode():
        outputs: Any = model_callable(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    logits = cast(torch.Tensor, outputs.logits)

    return logits[0, -1, :]


def run_packed_prefill_logits(
    model: PreTrainedModel,
    layout: BatchLayout,
    *,
    device: torch.device,
    mask_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run multiple prefills in one packed model forward.

    Args:
        model: Causal language model under test.
        layout: Packed layout containing only prefill executions.
        device: Device on which model inputs should be created.
        mask_dtype: Floating-point dtype used for the model-facing mask.

    Returns:
        One next-token logit vector per execution with shape
        ``(num_executions, vocab_size)``.

    Raises:
        ValueError: If the layout contains a non-prefill execution.
    """
    if any(
        execution.phase is not ExecutionPhase.PREFILL for execution in layout.executions
    ):
        raise ValueError("Packed prefill probe supports PREFILL executions only.")

    packed = build_packed_inputs(
        layout,
        device=device,
    )

    semantic_mask = build_attention_mask(
        layout,
        device=device,
    )

    model_attention_mask = semantic_to_additive_attention_mask(
        semantic_mask,
        dtype=mask_dtype,
    )

    input_ids = packed.input_ids.unsqueeze(0)
    position_ids = packed.position_ids.unsqueeze(0)

    model_callable: Any = model

    with torch.inference_mode():
        outputs: Any = model_callable(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=model_attention_mask,
            use_cache=False,
        )

    logits = cast(torch.Tensor, outputs.logits)

    logit_indices = torch.tensor(
        layout.logit_indices,
        dtype=torch.long,
        device=device,
    )

    return logits[0].index_select(
        dim=0,
        index=logit_indices,
    )


def run_packed_mixed_logits(
    model: PreTrainedModel,
    layout: BatchLayout,
    prefix_layers: tuple[LayerKV, ...],
    *,
    device: torch.device,
    mask_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one decode and one prefill in a single packed Qwen forward.

    The function also captures the mask received by Qwen's first attention
    layer so the probe verifies that Transformers did not replace or alter the
    explicit 4-D mask.

    Args:
        model: Causal language model under test.
        layout: Layout containing one decode followed by one prefill.
        prefix_layers: Cached per-layer K/V for the decode request's history
            before its newest query token.
        device: Device on which model inputs should be created.
        mask_dtype: Floating-point dtype used for the model-facing mask.

    Returns:
        Tuple containing gathered per-execution logits, the explicit mask passed
        to Qwen, and the mask observed by the first attention layer.

    Raises:
        ValueError: If the layout does not contain exactly DECODE then PREFILL.
        RuntimeError: If the first attention layer does not receive a tensor
            attention mask.
    """
    if len(layout.executions) != 2:
        raise ValueError("Mixed probe requires exactly two executions.")

    decode_execution, prefill_execution = layout.executions

    if decode_execution.phase is not ExecutionPhase.DECODE:
        raise ValueError("First mixed-probe execution must be DECODE.")

    if prefill_execution.phase is not ExecutionPhase.PREFILL:
        raise ValueError("Second mixed-probe execution must be PREFILL.")

    packed = build_packed_inputs(
        layout,
        device=device,
    )

    semantic_mask = build_attention_mask(
        layout,
        device=device,
    )

    model_attention_mask = semantic_to_additive_attention_mask(
        semantic_mask,
        dtype=mask_dtype,
    )

    input_ids = packed.input_ids.unsqueeze(0)
    position_ids = packed.position_ids.unsqueeze(0)

    cache = ProbeMixedCache(
        prefix_layers,
        decode_query_length=layout.query_lengths[0],
        prefill_query_length=layout.query_lengths[1],
        expected_key_tokens=layout.total_key_tokens,
    )

    model_any: Any = model
    first_attention = cast(
        torch.nn.Module,
        model_any.model.layers[0].self_attn,
    )

    captured_masks: list[torch.Tensor] = []

    def capture_mask(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        attention_mask = kwargs.get("attention_mask")

        if isinstance(attention_mask, torch.Tensor):
            captured_masks.append(attention_mask.detach().clone())

    hook_handle = first_attention.register_forward_pre_hook(
        capture_mask,
        with_kwargs=True,
    )

    model_callable: Any = model

    try:
        with torch.inference_mode():
            outputs: Any = model_callable(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=model_attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
    finally:
        hook_handle.remove()

    if not captured_masks:
        raise RuntimeError(
            "First Qwen attention layer did not receive a tensor attention mask."
        )

    logits = cast(torch.Tensor, outputs.logits)

    logit_indices = torch.tensor(
        layout.logit_indices,
        dtype=torch.long,
        device=device,
    )

    gathered_logits = logits[0].index_select(
        dim=0,
        index=logit_indices,
    )

    return (
        gathered_logits,
        model_attention_mask,
        captured_masks[0],
    )


def compare_logits(
    independent_logits: torch.Tensor,
    packed_logits: torch.Tensor,
) -> ProbeComparison:
    """Compare packed logits with independently produced reference logits.

    Args:
        independent_logits: Reference logits with shape
            ``(num_executions, vocab_size)``.
        packed_logits: Packed logits with the same shape.

    Returns:
        Top-1 and maximum-logit-difference comparison.

    Raises:
        ValueError: If the two logit tensors have different shapes.
    """
    if independent_logits.shape != packed_logits.shape:
        raise ValueError(
            "Independent and packed logits must have equal shapes: "
            f"{tuple(independent_logits.shape)} != {tuple(packed_logits.shape)}."
        )

    independent_top1 = tuple(
        int(token_id) for token_id in independent_logits.argmax(dim=-1).tolist()
    )

    packed_top1 = tuple(
        int(token_id) for token_id in packed_logits.argmax(dim=-1).tolist()
    )

    max_abs_logit_diffs = tuple(
        float((packed_logits[index] - independent_logits[index]).abs().max().item())
        for index in range(independent_logits.shape[0])
    )

    return ProbeComparison(
        independent_top1=independent_top1,
        packed_top1=packed_top1,
        max_abs_logit_diffs=max_abs_logit_diffs,
    )


def compare_prefill_batch(
    model: PreTrainedModel,
    layout: BatchLayout,
    *,
    device: torch.device,
) -> ProbeComparison:
    """Compare packed prefills against independent references.

    Args:
        model: Causal language model under test.
        layout: Packed layout containing prefill executions.
        device: Device on which probe execution should run.

    Returns:
        Comparison containing top-1 tokens and per-execution logit differences.
    """
    independent_logits = torch.stack(
        [
            run_independent_prefill_logits(
                model,
                execution,
                device=device,
            )
            for execution in layout.executions
        ]
    )

    packed_logits = run_packed_prefill_logits(
        model,
        layout,
        device=device,
    )

    return compare_logits(
        independent_logits,
        packed_logits,
    )


def compare_mixed_batch(
    model: PreTrainedModel,
    layout: BatchLayout,
    *,
    device: torch.device,
) -> MixedProbeResult:
    """Compare mixed decode-plus-prefill packed execution with references.

    Args:
        model: Causal language model under test.
        layout: Layout containing one decode followed by one prefill.
        device: Device on which probe execution should run.

    Returns:
        Mixed-probe comparison and mask-preservation evidence.

    Raises:
        ValueError: If the layout does not contain DECODE followed by PREFILL.
    """
    if len(layout.executions) != 2:
        raise ValueError("Mixed comparison requires exactly two executions.")

    decode_execution, prefill_execution = layout.executions

    if decode_execution.phase is not ExecutionPhase.DECODE:
        raise ValueError("Mixed comparison requires DECODE first.")

    if prefill_execution.phase is not ExecutionPhase.PREFILL:
        raise ValueError("Mixed comparison requires PREFILL second.")

    decode_history = decode_execution.sequence.all_token_ids

    prefix_token_ids = decode_history[:-1]
    decode_token_id = decode_history[-1]

    prefix_cache, prefix_layers = build_decode_prefix_cache(
        model,
        prefix_token_ids,
        device=device,
    )

    decode_reference = run_independent_decode_logits(
        model,
        prefix_cache,
        token_id=decode_token_id,
        logical_position=decode_execution.sequence.current_length - 1,
        device=device,
    )

    prefill_reference = run_independent_prefill_logits(
        model,
        prefill_execution,
        device=device,
    )

    independent_logits = torch.stack(
        (
            decode_reference,
            prefill_reference,
        )
    )

    (
        packed_logits,
        model_attention_mask,
        received_attention_mask,
    ) = run_packed_mixed_logits(
        model,
        layout,
        prefix_layers,
        device=device,
    )

    comparison = compare_logits(
        independent_logits,
        packed_logits,
    )

    return MixedProbeResult(
        comparison=comparison,
        model_mask_shape=tuple(model_attention_mask.shape),
        received_mask_shape=tuple(received_attention_mask.shape),
        mask_preserved=torch.equal(
            model_attention_mask,
            received_attention_mask,
        ),
    )


def print_layout(
    title: str,
    layout: BatchLayout,
    *,
    device: torch.device,
) -> None:
    """Print packed geometry and tensor shapes for one probe case.

    Args:
        title: Human-readable probe case name.
        layout: Packed batch geometry to describe.
        device: Device on which packed tensors should be constructed.
    """
    packed = build_packed_inputs(
        layout,
        device=device,
    )

    semantic_mask = build_attention_mask(
        layout,
        device=device,
    )

    model_attention_mask = semantic_to_additive_attention_mask(
        semantic_mask,
        dtype=torch.float32,
    )

    print(f"=== {title} ===")
    print(f"query lengths:       {layout.query_lengths}")
    print(f"key lengths:         {layout.key_lengths}")
    print(f"query offsets:       {layout.query_offsets}")
    print(f"key offsets:         {layout.key_offsets}")
    print(f"logit indices:       {layout.logit_indices}")
    print(f"input_ids:           {packed.input_ids.tolist()}")
    print(f"position_ids:        {packed.position_ids.tolist()}")
    print(f"input_ids shape:     {tuple(packed.input_ids.shape)}")
    print(f"position_ids shape:  {tuple(packed.position_ids.shape)}")
    print(f"semantic mask shape: {tuple(semantic_mask.shape)}")
    print(f"model mask shape:    {tuple(model_attention_mask.shape)}")
    print()


def print_comparison(comparison: ProbeComparison) -> None:
    """Print one independent-versus-packed comparison.

    Args:
        comparison: Comparison result to display.
    """
    print(f"independent top-1: {comparison.independent_top1}")
    print(f"packed top-1:      {comparison.packed_top1}")
    print(f"max logit diffs:   {comparison.max_abs_logit_diffs}")
    print(f"top-1 match:       {comparison.all_top1_match}")


def main() -> None:
    """Run packed-prefill and mixed-cache Qwen compatibility probes on CPU."""
    device = torch.device("cpu")

    prefill_layout = BatchLayout(
        executions=(
            make_prefill_execution(
                request_id="prefill-a",
                token_ids=(10, 11, 12),
                slot_start=0,
            ),
            make_prefill_execution(
                request_id="prefill-b",
                token_ids=(20, 21, 22, 23, 24),
                slot_start=16,
            ),
        )
    )

    mixed_layout = BatchLayout(
        executions=(
            make_decode_execution(
                request_id="decode-a",
                prompt_token_ids=(10, 11, 12),
                generated_token_ids=(13,),
                slot_start=0,
            ),
            make_prefill_execution(
                request_id="prefill-b",
                token_ids=(20, 21, 22),
                slot_start=16,
            ),
        )
    )

    print("=== Packed Qwen compatibility probe ===")
    print(f"device: {device}")
    print()

    print_layout(
        "9a.4a unequal PREFILL + PREFILL",
        prefill_layout,
        device=device,
    )

    print_layout(
        "9a.4b DECODE + PREFILL",
        mixed_layout,
        device=device,
    )

    print("Loading Qwen...")
    model = load_qwen_for_probe(device)
    print()

    print("=== 9a.4a results ===")

    prefill_comparison = compare_prefill_batch(
        model,
        prefill_layout,
        device=device,
    )

    print_comparison(prefill_comparison)

    if not prefill_comparison.all_top1_match:
        raise RuntimeError(
            "Packed unequal-prefill top-1 does not match independent execution."
        )

    print()
    print("=== 9a.4b results ===")

    mixed_result = compare_mixed_batch(
        model,
        mixed_layout,
        device=device,
    )

    print_comparison(mixed_result.comparison)

    print(f"model mask shape:    {mixed_result.model_mask_shape}")
    print(f"received mask shape: {mixed_result.received_mask_shape}")
    print(f"mask preserved:      {mixed_result.mask_preserved}")

    if not mixed_result.mask_preserved:
        raise RuntimeError(
            "Qwen attention did not receive the explicit packed mask unchanged."
        )

    if not mixed_result.comparison.all_top1_match:
        raise RuntimeError(
            "Packed mixed-phase top-1 does not match independent execution."
        )

    print()
    print("All packed Qwen compatibility probes passed.")


if __name__ == "__main__":
    main()
