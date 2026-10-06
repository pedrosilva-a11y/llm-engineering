"""Immutable benchmark prompt-set loading."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkPromptSet:
    """Versioned collection of tokenized benchmark prompts.

    Attributes:
        prompt_set_id: Stable identifier for the prompt set.
        model_name: Model/tokenizer identifier used to create the prompts.
        model_revision: Exact tokenizer revision used to create the prompts.
        prompt_length: Required token count for every prompt.
        prompt_token_ids_by_request: Tokenized prompts in deterministic order.
        sha256: SHA-256 digest of the prompt-set file contents.
    """

    prompt_set_id: str
    model_name: str
    model_revision: str
    prompt_length: int
    prompt_token_ids_by_request: tuple[tuple[int, ...], ...]
    sha256: str


def load_prompt_set(path: Path) -> BenchmarkPromptSet:
    """Load and validate one immutable benchmark prompt set.

    Args:
        path: Path to the JSON file containing the versioned benchmark prompt set.

    Returns:
        Validated benchmark prompt set containing its metadata, tokenized prompts
        in deterministic order, and SHA-256 digest of the source file contents.

    Raises:
        OSError: If the prompt-set file cannot be read.
        ValueError: If the file does not contain valid JSON prompt-set data, required
            metadata is missing or invalid, prompts are missing or malformed, prompt
            lengths do not match the declared length, token IDs are invalid, or
            duplicate prompts are present.
    """
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)

    if not isinstance(payload, dict):
        raise ValueError("prompt set must contain a JSON object.")

    prompt_set_id = _require_string(payload, "prompt_set_id")
    model_name = _require_string(payload, "model_name")
    model_revision = _require_string(payload, "model_revision")

    prompt_length = payload.get("prompt_length")
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("prompt_length must be a positive integer.")

    raw_prompts = payload.get("prompt_token_ids_by_request")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError(
            "prompt_token_ids_by_request must be a non-empty list.",
        )

    prompts: list[tuple[int, ...]] = []

    for raw_prompt in raw_prompts:
        if not isinstance(raw_prompt, list):
            raise ValueError("each benchmark prompt must be a list of token IDs.")

        if len(raw_prompt) != prompt_length:
            raise ValueError(
                f"each benchmark prompt must contain exactly {prompt_length} tokens.",
            )

        if any(type(token_id) is not int or token_id < 0 for token_id in raw_prompt):
            raise ValueError(
                "benchmark prompts must contain only non-negative integer token IDs.",
            )

        prompts.append(tuple(raw_prompt))

    prompt_tuple = tuple(prompts)

    if len(set(prompt_tuple)) != len(prompt_tuple):
        raise ValueError("benchmark prompts must be distinct.")

    return BenchmarkPromptSet(
        prompt_set_id=prompt_set_id,
        model_name=model_name,
        model_revision=model_revision,
        prompt_length=prompt_length,
        prompt_token_ids_by_request=prompt_tuple,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    """Read one required non-empty string field.

    Args:
        payload: Mapping containing the field to read.
        field_name: Name of the required field.

    Returns:
        The non-empty string value associated with ``field_name``.

    Raises:
        ValueError: If the field is missing, is not a string, or is empty.
    """
    value = payload.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    return value
