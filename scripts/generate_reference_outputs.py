"""Generate Day 7 greedy reference outputs with the naive model runner."""

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.naive_model_runner import NaiveModelRunner
from serving.engine.request import Request
from serving.engine.sampling import SamplingParameters

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

BLOCK_SIZE = 16
EXPECTED_EOS_TOKEN_ID = 151645

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reference_outputs" / "day7_qwen2_5_1_5b.json"


@dataclass(frozen=True)
class ReferenceCase:
    """Describe one deterministic reference-generation case."""

    name: str
    prompt: str
    max_new_tokens: int
    expected_finish_reason: str | None = None


def render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[str, tuple[int, ...]]:
    """Apply the Qwen chat template and tokenize the resulting prompt."""
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    )

    token_ids = tokenizer.encode(
        rendered,
        add_special_tokens=False,
    )

    return rendered, tuple(token_ids)


def find_prompt_with_remainder(
    tokenizer: PreTrainedTokenizerBase,
    base_prompt: str,
    remainder: int,
) -> str:
    """Find a prompt whose templated token count has the requested remainder."""
    for repetitions in range(256):
        candidate = base_prompt + (" x" * repetitions)

        _, token_ids = render_prompt(
            tokenizer=tokenizer,
            prompt=candidate,
        )

        if len(token_ids) % BLOCK_SIZE == remainder:
            return candidate

    raise RuntimeError(f"Could not find prompt with block remainder {remainder}.")


def build_reference_cases(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[ReferenceCase, ...]:
    """Build deterministic reference cases for important engine boundaries."""
    exact_boundary_prompt = find_prompt_with_remainder(
        tokenizer=tokenizer,
        base_prompt="Continue this sentence briefly:",
        remainder=0,
    )

    crossing_prompt = find_prompt_with_remainder(
        tokenizer=tokenizer,
        base_prompt="Continue with several words:",
        remainder=BLOCK_SIZE - 1,
    )

    return (
        ReferenceCase(
            name="natural_eos",
            prompt="Explain recursion in one short sentence.",
            max_new_tokens=32,
            expected_finish_reason="eos",
        ),
        ReferenceCase(
            name="length_termination",
            prompt=(
                "Write a detailed explanation of how transformer "
                "language models perform autoregressive decoding."
            ),
            max_new_tokens=4,
            expected_finish_reason="length",
        ),
        ReferenceCase(
            name="exact_block_boundary",
            prompt=exact_boundary_prompt,
            max_new_tokens=8,
        ),
        ReferenceCase(
            name="decode_crosses_block_boundary",
            prompt=crossing_prompt,
            max_new_tokens=8,
        ),
        ReferenceCase(
            name="long_multi_block_prompt",
            prompt=(
                "Explain the relationship between attention, key-value "
                "caching, autoregressive decoding, and GPU memory usage "
                "in a transformer inference server."
            ),
            max_new_tokens=12,
        ),
    )


def git_sha() -> str:
    """Return the repository commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def git_is_dirty() -> bool:
    """Return whether the repository has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return bool(result.stdout.strip())


def main() -> None:
    """Generate and persist the Day 7 reference outputs."""
    if not torch.cuda.is_available():
        raise RuntimeError("Reference generation requires a CUDA GPU.")

    if git_is_dirty():
        raise RuntimeError("Reference generation requires a clean Git working tree.")

    device = torch.device("cuda")

    tokenizer_factory: Any = AutoTokenizer
    tokenizer = cast(
        PreTrainedTokenizerBase,
        tokenizer_factory.from_pretrained(
            MODEL_ID,
            revision=REVISION,
        ),
    )

    eos_token_id = tokenizer.eos_token_id

    if eos_token_id != EXPECTED_EOS_TOKEN_ID:
        raise RuntimeError(
            "Unexpected EOS token ID: "
            f"expected {EXPECTED_EOS_TOKEN_ID}, got {eos_token_id}."
        )

    model_factory: Any = AutoModelForCausalLM
    loaded_model: Any = model_factory.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float16,
    )
    loaded_model = loaded_model.to(device)

    model = cast(PreTrainedModel, loaded_model)

    runner = NaiveModelRunner(
        model=model,
        device=device,
    )

    engine = Engine(
        configuration=EngineConfiguration(
            device="cuda",
            max_sequences=1,
            eos_token_id=EXPECTED_EOS_TOKEN_ID,
            block_size=BLOCK_SIZE,
            num_blocks=256,
            max_batched_tokens=2_048,
            seed=0,
        ),
        model_runner=runner,
    )

    results: list[dict[str, Any]] = []

    for case in build_reference_cases(tokenizer):
        rendered_prompt, prompt_token_ids = render_prompt(
            tokenizer=tokenizer,
            prompt=case.prompt,
        )

        request = Request(
            request_id=case.name,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=case.max_new_tokens,
            sampling_parameters=SamplingParameters(
                greedy=True,
            ),
        )

        sequence = engine.submit(request)
        engine.run_until_complete()

        if sequence.finish_reason is None:
            raise RuntimeError(f"Reference case {case.name!r} did not finish.")

        finish_reason = sequence.finish_reason.value

        if (
            case.expected_finish_reason is not None
            and finish_reason != case.expected_finish_reason
        ):
            raise RuntimeError(
                f"Case {case.name!r} finished with "
                f"{finish_reason!r}, expected "
                f"{case.expected_finish_reason!r}."
            )

        generated_token_ids = list(sequence.generated_token_ids)

        results.append(
            {
                "name": case.name,
                "prompt": case.prompt,
                "rendered_prompt": rendered_prompt,
                "prompt_token_count": len(prompt_token_ids),
                "prompt_block_remainder": (len(prompt_token_ids) % BLOCK_SIZE),
                "prompt_token_ids": list(prompt_token_ids),
                "max_new_tokens": case.max_new_tokens,
                "generated_token_ids": generated_token_ids,
                "generated_token_count": len(generated_token_ids),
                "finish_reason": finish_reason,
                "decoded_text": tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=True,
                ),
            }
        )

    artifact = {
        "metadata": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "dtype": "float16",
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "git_sha": git_sha(),
            "block_size": BLOCK_SIZE,
            "eos_token_id": EXPECTED_EOS_TOKEN_ID,
            "seed": 0,
            "sampling": {
                "greedy": True,
            },
            "chat_template": {
                "tokenizer_default": True,
                "add_generation_prompt": True,
            },
        },
        "cases": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(results)} reference cases.")
    print(f"Output: {OUTPUT_PATH}")

    for result in results:
        print(
            f"- {result['name']}: "
            f"prompt={result['prompt_token_count']} tokens, "
            f"generated={result['generated_token_count']}, "
            f"finish={result['finish_reason']}"
        )


if __name__ == "__main__":
    main()
