"""Validate paged KV inference against frozen Day 7 reference outputs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.paged_model_runner import PagedModelRunner
from serving.engine.request import Request
from serving.engine.sampling import SamplingParameters

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

DTYPE = torch.float16
BLOCK_SIZE = 16
NUM_BLOCKS = 256
EXPECTED_EOS_TOKEN_ID = 151645
SEED = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "reference_outputs" / "day7_qwen2_5_1_5b.json"


@dataclass(frozen=True)
class ReferenceMetadata:
    """Describe the frozen inference configuration."""

    model_id: str
    revision: str
    dtype: str
    gpu: str
    transformers_version: str
    block_size: int
    eos_token_id: int
    seed: int


@dataclass(frozen=True)
class ReferenceCase:
    """Describe one frozen generation result."""

    name: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    generated_token_ids: tuple[int, ...]
    finish_reason: str


def load_reference() -> tuple[ReferenceMetadata, tuple[ReferenceCase, ...]]:
    """Load frozen Day 7 metadata and generation cases."""
    raw: Any = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    metadata_raw = cast(dict[str, Any], raw["metadata"])
    cases_raw = cast(list[dict[str, Any]], raw["cases"])

    metadata = ReferenceMetadata(
        model_id=str(metadata_raw["model_id"]),
        revision=str(metadata_raw["revision"]),
        dtype=str(metadata_raw["dtype"]),
        gpu=str(metadata_raw["gpu"]),
        transformers_version=str(metadata_raw["transformers_version"]),
        block_size=int(metadata_raw["block_size"]),
        eos_token_id=int(metadata_raw["eos_token_id"]),
        seed=int(metadata_raw["seed"]),
    )

    cases = tuple(
        ReferenceCase(
            name=str(case["name"]),
            prompt_token_ids=tuple(
                int(token_id)
                for token_id in cast(
                    list[int],
                    case["prompt_token_ids"],
                )
            ),
            max_new_tokens=int(case["max_new_tokens"]),
            generated_token_ids=tuple(
                int(token_id)
                for token_id in cast(
                    list[int],
                    case["generated_token_ids"],
                )
            ),
            finish_reason=str(case["finish_reason"]),
        )
        for case in cases_raw
    )

    return metadata, cases


def validate_reference_metadata(
    metadata: ReferenceMetadata,
) -> None:
    """Validate the frozen artifact against the expected Day 7 contract."""
    expected_values: tuple[tuple[str, object, object], ...] = (
        ("model_id", metadata.model_id, MODEL_ID),
        ("revision", metadata.revision, REVISION),
        ("dtype", metadata.dtype, "float16"),
        ("block_size", metadata.block_size, BLOCK_SIZE),
        (
            "eos_token_id",
            metadata.eos_token_id,
            EXPECTED_EOS_TOKEN_ID,
        ),
        ("seed", metadata.seed, SEED),
        (
            "transformers_version",
            metadata.transformers_version,
            transformers.__version__,
        ),
    )

    for name, actual, expected in expected_values:
        if actual != expected:
            raise RuntimeError(
                f"Reference metadata mismatch for {name}: "
                f"expected {expected!r}, got {actual!r}."
            )


def model_geometry(
    model: PreTrainedModel,
) -> tuple[int, int, int]:
    """Return transformer layer, KV-head, and head-dimension counts."""
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_kv_heads = int(getattr(model.config, "num_key_value_heads"))

    configured_head_dim = getattr(
        model.config,
        "head_dim",
        None,
    )

    if configured_head_dim is not None:
        head_dim = int(configured_head_dim)
    else:
        hidden_size = int(getattr(model.config, "hidden_size"))
        num_attention_heads = int(getattr(model.config, "num_attention_heads"))
        head_dim = hidden_size // num_attention_heads

    return num_layers, num_kv_heads, head_dim


def validate_generated_tokens(
    case: ReferenceCase,
    actual_token_ids: tuple[int, ...],
) -> None:
    """Require generated tokens to exactly match the frozen reference."""
    expected_token_ids = case.generated_token_ids
    common_length = min(
        len(expected_token_ids),
        len(actual_token_ids),
    )

    for index in range(common_length):
        expected = expected_token_ids[index]
        actual = actual_token_ids[index]

        if actual != expected:
            raise RuntimeError(
                f"Case {case.name!r} diverged at generated token "
                f"{index}: expected {expected}, got {actual}."
            )

    if len(actual_token_ids) != len(expected_token_ids):
        mismatch_index = common_length

        expected_value: int | str
        actual_value: int | str

        if mismatch_index < len(expected_token_ids):
            expected_value = expected_token_ids[mismatch_index]
        else:
            expected_value = "<end>"

        if mismatch_index < len(actual_token_ids):
            actual_value = actual_token_ids[mismatch_index]
        else:
            actual_value = "<end>"

        raise RuntimeError(
            f"Case {case.name!r} diverged at generated token "
            f"{mismatch_index}: expected {expected_value}, "
            f"got {actual_value}."
        )


def main() -> None:
    """Validate paged inference against every frozen Day 7 case."""
    if not torch.cuda.is_available():
        raise RuntimeError("Reference validation requires a CUDA GPU.")

    metadata, cases = load_reference()
    validate_reference_metadata(metadata)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    print("=== Frozen reference ===")
    print(f"Model: {metadata.model_id}")
    print(f"Revision: {metadata.revision}")
    print(f"Reference GPU: {metadata.gpu}")
    print(f"Current GPU: {gpu_name}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Cases: {len(cases)}")

    model_factory: Any = AutoModelForCausalLM
    loaded_model: Any = model_factory.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=DTYPE,
    )
    loaded_model = loaded_model.to(device)

    model = cast(PreTrainedModel, loaded_model)

    num_layers, num_kv_heads, head_dim = model_geometry(model)

    storage = PagedKVStorage(
        num_layers=num_layers,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=DTYPE,
        device=device,
    )

    paged_cache = PagedKVCache(storage=storage)

    runner = PagedModelRunner(
        model=model,
        device=device,
        paged_cache=paged_cache,
    )

    engine = Engine(
        configuration=EngineConfiguration(
            device="cuda",
            max_sequences=1,
            eos_token_id=EXPECTED_EOS_TOKEN_ID,
            block_size=BLOCK_SIZE,
            num_blocks=NUM_BLOCKS,
            max_batched_tokens=2_048,
            seed=SEED,
        ),
        model_runner=runner,
    )

    print()
    print("=== Paged KV configuration ===")
    print(f"Layers: {num_layers}")
    print(f"KV heads: {num_kv_heads}")
    print(f"Head dimension: {head_dim}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Blocks: {NUM_BLOCKS}")
    print(f"Physical slots: {storage.num_slots}")

    print()
    print("=== Reference validation ===")

    for case in cases:
        request = Request(
            request_id=case.name,
            prompt_token_ids=case.prompt_token_ids,
            max_new_tokens=case.max_new_tokens,
            sampling_parameters=SamplingParameters(
                greedy=True,
            ),
        )

        sequence = engine.submit(request)
        engine.run_until_complete()

        if sequence.finish_reason is None:
            raise RuntimeError(f"Case {case.name!r} did not finish.")

        actual_token_ids = tuple(sequence.generated_token_ids)

        validate_generated_tokens(
            case=case,
            actual_token_ids=actual_token_ids,
        )

        actual_finish_reason = sequence.finish_reason.value

        if actual_finish_reason != case.finish_reason:
            raise RuntimeError(
                f"Case {case.name!r} finish reason mismatch: "
                f"expected {case.finish_reason!r}, "
                f"got {actual_finish_reason!r}."
            )

        print(
            f"PASS {case.name}: "
            f"prompt={len(case.prompt_token_ids)}, "
            f"generated={len(actual_token_ids)}, "
            f"finish={actual_finish_reason}"
        )

    print()
    print(f"PASS: all {len(cases)} frozen Day 7 reference cases matched exactly.")


if __name__ == "__main__":
    main()
