"""Integration tests for the real Qwen tokenizer."""

import os

import pytest

from serving.gateway.tokenizer import QwenTextTokenizer

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_TOKENIZER_INTEGRATION") != "1",
    reason="real tokenizer integration test is opt-in",
)


@pytest.fixture(scope="module")
def tokenizer() -> QwenTextTokenizer:
    """Load the pinned real Qwen tokenizer once for integration tests."""
    return QwenTextTokenizer.from_pretrained()


def test_real_qwen_tokenizer_encodes_prompt(tokenizer: QwenTextTokenizer) -> None:
    """Encode prompt with the pinned real Qwen tokenizer."""
    token_ids = tokenizer.encode_prompt("Hello")

    assert isinstance(token_ids, tuple)
    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_real_qwen_tokenizer_decodes_empty_sequence(
    tokenizer: QwenTextTokenizer,
) -> None:
    """Decode an empty generated-token sequence as empty text."""
    assert tokenizer.decode([]) == ""
