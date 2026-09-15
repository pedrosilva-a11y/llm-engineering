"""Measure prompt KV-cache fragmentation for frozen Day 7 cases."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from serving.engine.config import EngineConfiguration
from serving.engine.request import Request
from serving.engine.scheduler import Scheduler

BLOCK_SIZE = 16
NUM_BLOCKS = 256
EOS_TOKEN_ID = 151645

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "reference_outputs" / "day7_qwen2_5_1_5b.json"


@dataclass(frozen=True)
class FragmentationCase:
    """Describe one prompt used for KV-cache fragmentation measurement."""

    name: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class FragmentationResult:
    """Describe KV-cache fragmentation for one admitted prompt."""

    name: str
    prompt_tokens: int
    allocated_blocks: int
    allocated_slots: int
    wasted_slots: int

    @property
    def utilization(self) -> float:
        """Return the fraction of allocated slots occupied by prompt tokens."""
        return self.prompt_tokens / self.allocated_slots


def load_cases() -> tuple[FragmentationCase, ...]:
    """Load frozen prompt cases from the Day 7 reference artifact."""
    raw: Any = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    cases_raw = cast(list[dict[str, Any]], raw["cases"])

    return tuple(
        FragmentationCase(
            name=str(case["name"]),
            prompt_token_ids=tuple(
                int(token_id)
                for token_id in cast(
                    list[int],
                    case["prompt_token_ids"],
                )
            ),
            max_new_tokens=int(case["max_new_tokens"]),
        )
        for case in cases_raw
    )


def measure_case(case: FragmentationCase) -> FragmentationResult:
    """Measure KV fragmentation immediately after scheduler admission."""
    scheduler = Scheduler(
        configuration=EngineConfiguration(
            device="cpu",
            max_sequences=1,
            eos_token_id=EOS_TOKEN_ID,
            block_size=BLOCK_SIZE,
            num_blocks=NUM_BLOCKS,
            max_batched_tokens=2_048,
            seed=0,
        )
    )

    sequence = scheduler.submit(
        Request(
            request_id=case.name,
            prompt_token_ids=case.prompt_token_ids,
            max_new_tokens=case.max_new_tokens,
        )
    )

    scheduler_output = scheduler.schedule()

    if sequence not in scheduler_output.prefill_sequences:
        raise RuntimeError(f"Case {case.name!r} was not admitted for prefill.")

    prompt_tokens = len(case.prompt_token_ids)

    wasted_slots = scheduler.block_table.fragmentation_slots(
        request_id=case.name,
        num_tokens=prompt_tokens,
    )

    allocated_slots = prompt_tokens + wasted_slots
    allocated_blocks = allocated_slots // BLOCK_SIZE

    return FragmentationResult(
        name=case.name,
        prompt_tokens=prompt_tokens,
        allocated_blocks=allocated_blocks,
        allocated_slots=allocated_slots,
        wasted_slots=wasted_slots,
    )


def main() -> None:
    """Measure and print prompt KV fragmentation for all frozen cases."""
    cases = load_cases()
    results = tuple(measure_case(case) for case in cases)

    print("=== Prompt KV-cache fragmentation ===")
    print(f"Block size: {BLOCK_SIZE}")
    print()

    for result in results:
        print(
            f"{result.name}: "
            f"tokens={result.prompt_tokens}, "
            f"blocks={result.allocated_blocks}, "
            f"slots={result.allocated_slots}, "
            f"wasted={result.wasted_slots}, "
            f"utilization={result.utilization:.2%}"
        )

    total_tokens = sum(result.prompt_tokens for result in results)
    total_slots = sum(result.allocated_slots for result in results)
    total_wasted = sum(result.wasted_slots for result in results)

    print()
    print("=== Aggregate ===")
    print(f"Prompt tokens: {total_tokens}")
    print(f"Allocated slots: {total_slots}")
    print(f"Wasted slots: {total_wasted}")
    print(f"Slot utilization: {total_tokens / total_slots:.2%}")


if __name__ == "__main__":
    main()
