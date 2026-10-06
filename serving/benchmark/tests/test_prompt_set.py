"""Tests for immutable benchmark prompt sets."""

import hashlib
import json
from pathlib import Path

import pytest

from serving.benchmark.prompt_set import load_prompt_set


def test_load_prompt_set_preserves_prompt_order(tmp_path: Path) -> None:
    """Load distinct fixed-length prompts in deterministic order."""
    path = tmp_path / "prompts.json"

    path.write_text(
        json.dumps(
            {
                "prompt_set_id": "test-v1",
                "model_name": "test-model",
                "model_revision": "abc123",
                "prompt_length": 3,
                "prompt_token_ids_by_request": [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            }
        )
    )

    prompt_set = load_prompt_set(path)

    assert prompt_set.prompt_set_id == "test-v1"
    assert prompt_set.model_name == "test-model"
    assert prompt_set.model_revision == "abc123"
    assert prompt_set.prompt_length == 3
    assert prompt_set.prompt_token_ids_by_request == (
        (1, 2, 3),
        (4, 5, 6),
    )
    assert prompt_set.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "prompts",
    (
        [],
        [[1, 2]],
        [[1, 2, 3], [1, 2, 3]],
        [[1, 2, -1]],
        [[1, 2, True]],
        [[1, 2, "3"]],
    ),
)
def test_load_prompt_set_rejects_invalid_prompts(
    tmp_path: Path,
    prompts: list[list[object]],
) -> None:
    """Reject malformed, wrong-length, duplicate, or invalid prompts."""
    path = tmp_path / "prompts.json"

    path.write_text(
        json.dumps(
            {
                "prompt_set_id": "test-v1",
                "model_name": "test-model",
                "model_revision": "abc123",
                "prompt_length": 3,
                "prompt_token_ids_by_request": prompts,
            }
        )
    )

    with pytest.raises(ValueError):
        load_prompt_set(path)


def test_load_prompt_set_rejects_non_object_payload(tmp_path: Path) -> None:
    """Reject a JSON root that is not an object."""
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="prompt set must contain a JSON object"):
        load_prompt_set(path)


@pytest.mark.parametrize(
    "prompt_length",
    (
        0,
        -1,
        True,
        "3",
    ),
)
def test_load_prompt_set_rejects_invalid_prompt_length(
    tmp_path: Path,
    prompt_length: object,
) -> None:
    """Reject non-positive and non-integer prompt lengths."""
    path = tmp_path / "prompts.json"

    path.write_text(
        json.dumps(
            {
                "prompt_set_id": "test-v1",
                "model_name": "test-model",
                "model_revision": "abc123",
                "prompt_length": prompt_length,
                "prompt_token_ids_by_request": [[1, 2, 3]],
            }
        )
    )

    with pytest.raises(ValueError, match="prompt_length must be a positive integer"):
        load_prompt_set(path)


def test_load_prompt_set_rejects_non_list_prompt(tmp_path: Path) -> None:
    """Reject a prompt that is not represented as a list."""
    path = tmp_path / "prompts.json"

    path.write_text(
        json.dumps(
            {
                "prompt_set_id": "test-v1",
                "model_name": "test-model",
                "model_revision": "abc123",
                "prompt_length": 3,
                "prompt_token_ids_by_request": ["not-a-prompt"],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="each benchmark prompt must be a list of token IDs",
    ):
        load_prompt_set(path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("prompt_set_id", ""),
        ("prompt_set_id", "   "),
        ("model_name", None),
        ("model_revision", 123),
    ),
)
def test_load_prompt_set_rejects_invalid_required_strings(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    """Reject missing, empty, or non-string required metadata."""
    payload: dict[str, object] = {
        "prompt_set_id": "test-v1",
        "model_name": "test-model",
        "model_revision": "abc123",
        "prompt_length": 3,
        "prompt_token_ids_by_request": [[1, 2, 3]],
    }
    payload[field_name] = invalid_value

    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=rf"{field_name} must be a non-empty string"):
        load_prompt_set(path)
