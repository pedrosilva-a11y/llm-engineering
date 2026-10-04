"""Tests for gateway text tokenization."""

from collections.abc import Sequence

import pytest
from transformers import AutoTokenizer

from serving.gateway.tokenizer import (
    QWEN_MODEL_NAME,
    QWEN_MODEL_REVISION,
    QwenTextTokenizer,
)


class _FakeTokenizerBackend:
    """Deterministic tokenizer backend used by the gateway tokenizer tests."""

    def __init__(
        self,
        *,
        encoded_token_ids: list[int] | None = None,
        decoded_text: str = "decoded text",
    ) -> None:
        """Initialize deterministic tokenizer behavior."""
        self.encoded_token_ids = (
            [10, 20, 30] if encoded_token_ids is None else encoded_token_ids
        )
        self.decoded_text = decoded_text

        self.chat_conversation: list[dict[str, str]] | None = None
        self.chat_tokenize: bool | None = None
        self.chat_add_generation_prompt: bool | None = None
        self.chat_return_dict: bool | None = None

        self.decoded_token_ids: tuple[int, ...] | None = None
        self.decode_skip_special_tokens: bool | None = None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        """Record chat-template arguments and return deterministic tokens."""
        self.chat_conversation = conversation
        self.chat_tokenize = tokenize
        self.chat_add_generation_prompt = add_generation_prompt
        self.chat_return_dict = return_dict

        return self.encoded_token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Record decode arguments and return deterministic text."""
        self.decoded_token_ids = tuple(token_ids)
        self.decode_skip_special_tokens = skip_special_tokens

        return self.decoded_text


def test_encode_prompt_applies_qwen_chat_template() -> None:
    """Apply the user chat template and request generation tokens."""
    backend = _FakeTokenizerBackend(encoded_token_ids=[101, 102, 103])
    tokenizer = QwenTextTokenizer(backend)

    token_ids = tokenizer.encode_prompt("Explain paged KV caching.")

    assert token_ids == (101, 102, 103)

    assert backend.chat_conversation == [
        {"role": "user", "content": "Explain paged KV caching."}
    ]
    assert backend.chat_tokenize is True
    assert backend.chat_add_generation_prompt is True
    assert backend.chat_return_dict is False


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        " ",
        "  ",
        "\n\t",
    ],
)
def test_encode_prompt_rejects_blank_prompt(prompt: str) -> None:
    """Reject empty or whitespace-only human-readable prompts."""
    backend = _FakeTokenizerBackend()
    tokenizer = QwenTextTokenizer(backend)

    with pytest.raises(ValueError, match="prompt must not be empty"):
        tokenizer.encode_prompt(prompt)

    assert backend.chat_conversation is None


def test_encode_prompt_rejects_empty_tokenization_result() -> None:
    """Reject a tokenizer result that contains no model tokens."""
    tokenizer = QwenTextTokenizer(_FakeTokenizerBackend(encoded_token_ids=[]))

    with pytest.raises(
        RuntimeError,
        match="chat-template tokenization produced no tokens",
    ):
        tokenizer.encode_prompt("Hello")


def test_decode_returns_cumulative_text_without_special_tokens() -> None:
    """Decode the complete generated prefix while removing special tokens."""
    backend = _FakeTokenizerBackend(decoded_text="Paged KV caching")
    tokenizer = QwenTextTokenizer(backend)

    text = tokenizer.decode([41, 42, 43])

    assert text == "Paged KV caching"
    assert backend.decoded_token_ids == (41, 42, 43)
    assert backend.decode_skip_special_tokens is True


def test_decode_accepts_empty_token_sequence() -> None:
    """Decode an empty generated-token sequence."""
    backend = _FakeTokenizerBackend(decoded_text="")
    tokenizer = QwenTextTokenizer(backend)

    text = tokenizer.decode([])

    assert text == ""
    assert backend.decoded_token_ids == ()
    assert backend.decode_skip_special_tokens is True


def test_from_pretrained_loads_pinned_qwen_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the exact Qwen model and revision frozen for the demo."""
    backend = _FakeTokenizerBackend()

    captured_model_name: str | None = None
    captured_revision: str | None = None
    captured_use_fast: bool | None = None

    def fake_from_pretrained(
        model_name: str,
        *,
        revision: str,
        use_fast: bool,
    ) -> _FakeTokenizerBackend:
        nonlocal captured_model_name
        nonlocal captured_revision
        nonlocal captured_use_fast

        captured_model_name = model_name
        captured_revision = revision
        captured_use_fast = use_fast

        return backend

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    tokenizer = QwenTextTokenizer.from_pretrained()

    assert isinstance(tokenizer, QwenTextTokenizer)
    assert captured_model_name == QWEN_MODEL_NAME
    assert captured_revision == QWEN_MODEL_REVISION
    assert captured_use_fast is True
