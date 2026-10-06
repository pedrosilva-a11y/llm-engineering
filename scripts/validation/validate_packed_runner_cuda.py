"""Validate packed paged execution against the naive runner on CUDA."""

from typing import Any, cast

import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.naive_model_runner import NaiveModelRunner
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.paged_model_runner import PagedModelRunner
from serving.engine.request import Request
from serving.engine.sequence import SequenceState

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

DTYPE = torch.bfloat16
BLOCK_SIZE = 16
NUM_BLOCKS = 8
TOTAL_SLOTS = NUM_BLOCKS * BLOCK_SIZE


def load_model(device: torch.device) -> PreTrainedModel:
    """Load the frozen Qwen checkpoint with the configured validation dtype and SDPA."""
    model_factory: Any = AutoModelForCausalLM

    loaded_model: Any = model_factory.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    loaded_model = loaded_model.to(device)

    model = cast(PreTrainedModel, loaded_model)
    cast(torch.nn.Module, model).eval()

    return model


def make_sequence(
    request_id: str,
    token_ids: tuple[int, ...],
) -> SequenceState:
    """Create a sequence from deterministic token identifiers."""
    return SequenceState(
        request=Request(
            request_id=request_id,
            prompt_token_ids=token_ids,
            max_new_tokens=8,
        )
    )


def make_execution(
    sequence: SequenceState,
    phase: ExecutionPhase,
    *,
    slot_start: int,
) -> ModelExecution:
    """Create execution metadata with deterministic physical slots."""
    end_slot = slot_start + sequence.current_length

    if end_slot > TOTAL_SLOTS:
        raise ValueError(
            f"Execution requires slots through {end_slot - 1}, "
            f"but storage only has {TOTAL_SLOTS} slots."
        )

    return ModelExecution(
        sequence=sequence,
        phase=phase,
        slot_ids=tuple(
            range(
                slot_start,
                end_slot,
            )
        ),
    )


def make_paged_runner(
    model: PreTrainedModel,
    device: torch.device,
) -> PagedModelRunner:
    """Create a fresh CUDA paged runner and KV storage."""
    configuration: Any = model.config

    head_dim_value = getattr(configuration, "head_dim", None)

    if head_dim_value is None:
        head_dim = int(configuration.hidden_size // configuration.num_attention_heads)
    else:
        head_dim = int(head_dim_value)

    storage = PagedKVStorage(
        num_layers=int(configuration.num_hidden_layers),
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=int(configuration.num_key_value_heads),
        head_dim=head_dim,
        dtype=DTYPE,
        device=device,
    )

    return PagedModelRunner(
        model=model,
        device=device,
        paged_cache=PagedKVCache(storage),
    )


def compare_logits(
    case_name: str,
    naive_logits: torch.Tensor,
    packed_logits: torch.Tensor,
) -> None:
    """Compare packed next-token predictions with the naive reference."""
    if naive_logits.shape != packed_logits.shape:
        raise AssertionError(
            f"{case_name}: shape mismatch: "
            f"naive={tuple(naive_logits.shape)}, "
            f"packed={tuple(packed_logits.shape)}"
        )

    naive_float = naive_logits.float()
    packed_float = packed_logits.float()

    if not torch.isfinite(naive_float).all():
        raise AssertionError(f"{case_name}: naive logits contain non-finite values.")

    if not torch.isfinite(packed_float).all():
        raise AssertionError(f"{case_name}: packed logits contain non-finite values.")

    max_abs_diffs = (naive_float - packed_float).abs().amax(dim=-1)

    naive_top1 = torch.argmax(naive_logits, dim=-1)
    packed_top1 = torch.argmax(packed_logits, dim=-1)

    naive_top2_values = torch.topk(
        naive_float,
        k=2,
        dim=-1,
    ).values

    packed_top2_values = torch.topk(
        packed_float,
        k=2,
        dim=-1,
    ).values

    naive_margins = naive_top2_values[:, 0] - naive_top2_values[:, 1]
    packed_margins = packed_top2_values[:, 0] - packed_top2_values[:, 1]

    print()
    print(f"=== {case_name} ===")

    for index in range(naive_logits.shape[0]):
        naive_token = int(naive_top1[index].item())
        packed_token = int(packed_top1[index].item())
        max_diff = float(max_abs_diffs[index].item())
        naive_margin = float(naive_margins[index].item())
        packed_margin = float(packed_margins[index].item())

        print(
            f"sequence {index}: "
            f"naive_top1={naive_token}, "
            f"packed_top1={packed_token}, "
            f"max_abs_diff={max_diff:.8f}, "
            f"naive_top2_margin={naive_margin:.8f}, "
            f"packed_top2_margin={packed_margin:.8f}"
        )

    if not torch.equal(naive_top1, packed_top1):
        raise AssertionError(
            f"{case_name}: packed top-1 predictions differ from naive reference."
        )

    print("top-1 match: PASS")


def validate_unequal_prefills(
    model: PreTrainedModel,
    device: torch.device,
    naive_runner: NaiveModelRunner,
) -> None:
    """Compare two unequal prefill executions."""
    paged_runner = make_paged_runner(model, device)

    first = make_sequence(
        request_id="prefill-a",
        token_ids=(10, 11, 12),
    )

    second = make_sequence(
        request_id="prefill-b",
        token_ids=(20, 21, 22, 23, 24),
    )

    executions = (
        make_execution(
            first,
            ExecutionPhase.PREFILL,
            slot_start=0,
        ),
        make_execution(
            second,
            ExecutionPhase.PREFILL,
            slot_start=32,
        ),
    )

    naive_logits = naive_runner.forward(executions)
    packed_logits = paged_runner.forward(executions)

    compare_logits(
        "9a.7a unequal PREFILL + PREFILL",
        naive_logits,
        packed_logits,
    )


def validate_unequal_decodes(
    model: PreTrainedModel,
    device: torch.device,
    naive_runner: NaiveModelRunner,
) -> None:
    """Compare two decode executions with unequal KV histories."""
    paged_runner = make_paged_runner(model, device)

    first = make_sequence(
        request_id="decode-a",
        token_ids=(10, 11, 12),
    )

    second = make_sequence(
        request_id="decode-b",
        token_ids=(20, 21, 22, 23, 24, 25),
    )

    prefill_executions = (
        make_execution(
            first,
            ExecutionPhase.PREFILL,
            slot_start=0,
        ),
        make_execution(
            second,
            ExecutionPhase.PREFILL,
            slot_start=32,
        ),
    )

    # Populate the real paged KV cache with model-produced prefix states.
    paged_runner.forward(prefill_executions)

    # Appending extends the logical history by one token. Rebuilding each
    # execution below preserves the existing prefix slots and adds exactly one
    # new final slot for the decode write.
    first.append_token(13)
    second.append_token(26)

    decode_executions = (
        make_execution(
            first,
            ExecutionPhase.DECODE,
            slot_start=0,
        ),
        make_execution(
            second,
            ExecutionPhase.DECODE,
            slot_start=32,
        ),
    )

    naive_logits = naive_runner.forward(decode_executions)
    packed_logits = paged_runner.forward(decode_executions)

    compare_logits(
        "9a.7b unequal DECODE + DECODE",
        naive_logits,
        packed_logits,
    )


def validate_mixed_decode_and_prefill(
    model: PreTrainedModel,
    device: torch.device,
    naive_runner: NaiveModelRunner,
) -> None:
    """Compare mixed decode and prefill execution."""
    paged_runner = make_paged_runner(model, device)

    decode_sequence = make_sequence(
        request_id="mixed-decode",
        token_ids=(10, 11, 12),
    )

    prefill_sequence = make_sequence(
        request_id="mixed-prefill",
        token_ids=(20, 21, 22),
    )

    # Establish only the decode sequence's previous KV history.
    paged_runner.forward(
        (
            make_execution(
                decode_sequence,
                ExecutionPhase.PREFILL,
                slot_start=0,
            ),
        )
    )

    # Rebuilding the decode execution after appending extends its slot mapping
    # from (0, 1, 2) to (0, 1, 2, 3). Only slot 3 should be written.
    decode_sequence.append_token(13)

    mixed_executions = (
        make_execution(
            decode_sequence,
            ExecutionPhase.DECODE,
            slot_start=0,
        ),
        make_execution(
            prefill_sequence,
            ExecutionPhase.PREFILL,
            slot_start=32,
        ),
    )

    naive_logits = naive_runner.forward(mixed_executions)
    packed_logits = paged_runner.forward(mixed_executions)

    compare_logits(
        "9a.7c DECODE + PREFILL",
        naive_logits,
        packed_logits,
    )


def main() -> None:
    """Run the CUDA packed-runner differential validation."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requires a CUDA GPU.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 validation requires a CUDA device with BF16 support.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda")

    print("=== 9a.7 packed paged runner CUDA validation ===")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {REVISION}")
    print(f"Dtype: {DTYPE}")
    print("Attention: sdpa")

    model = load_model(device)

    # NaiveModelRunner is stateless and recomputes complete histories, so one
    # instance can serve as the independent reference across all cases.
    # Both runners intentionally share the exact same model object.
    naive_runner = NaiveModelRunner(
        model=model,
        device=device,
    )

    validate_unequal_prefills(
        model,
        device,
        naive_runner,
    )

    validate_unequal_decodes(
        model,
        device,
        naive_runner,
    )

    validate_mixed_decode_and_prefill(
        model,
        device,
        naive_runner,
    )

    torch.cuda.synchronize()

    print()
    print("All packed CUDA differential validations passed.")


if __name__ == "__main__":
    main()
