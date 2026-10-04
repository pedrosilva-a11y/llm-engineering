"""Text tokenization utilities for the inference HTTP gateway."""

from collections.abc import Sequence
from typing import Protocol, Self, cast

from transformers import AutoTokenizer

QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class _TokenizerBackend(Protocol):
    """Minimal tokenizer interface required by the gateway."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        """Apply the model chat template to a conversation.

        Args:
            conversation: Ordered chat messages containing role and content fields.
            tokenize: Whether to tokenize the templated conversation.
            add_generation_prompt: Whether to append the model-specific generation
                marker indicating that an assistant response should begin.
            return_dict: Whether to return tokenizer outputs as a mapping instead of a
                flat sequence of token identifiers.

        Returns:
            Token identifiers produced from the templated conversation.
        """
        ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode token identifiers into human-readable text.

        Args:
            token_ids: Token identifiers to decode.
            skip_special_tokens: Whether model-specific special tokens should be omitted
                from the decoded output.

        Returns:
            Decoded text corresponding to the supplied token identifiers.
        """
        ...


class TextTokenizer(Protocol):
    """Convert between gateway text and model token identifiers."""

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        """Encode one human-readable prompt for model generation.

        Args:
            prompt: Human-readable prompt submitted through the gateway.

        Returns:
            Token identifiers ready to be passed to the inference engine. The encoded
            sequence may include model-specific template tokens in addition to tokens
            representing the raw prompt text.
        """
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated token identifiers into human-readable text.

        Args:
            token_ids: Generated token identifiers accumulated so far.

        Returns:
            Decoded text corresponding to the supplied token sequence.
        """
        ...


class QwenTextTokenizer:
    """Tokenize gateway text for the pinned Qwen instruct model."""

    def __init__(self, tokenizer: _TokenizerBackend) -> None:
        """Initialize the gateway tokenizer.

        Args:
            tokenizer: Underlying Hugging Face-compatible tokenizer.
        """
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load the tokenizer for the pinned Qwen model revision.

        Returns:
            Gateway tokenizer backed by the pinned Hugging Face tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,
            revision=QWEN_MODEL_REVISION,
            use_fast=True,
        )

        return cls(cast(_TokenizerBackend, tokenizer))

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        """Apply the Qwen chat template and tokenize one user prompt.

        Args:
            prompt: Human-readable user prompt.

        Returns:
            Token identifiers ready to be submitted to the inference engine. The
            sequence includes tokens introduced by the Qwen chat template and may
            therefore be substantially longer than the tokens corresponding to the
            raw prompt text alone.

        Raises:
            RuntimeError: If chat-template tokenization produces no tokens.
            ValueError: If the prompt is empty or contains only whitespace.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        token_ids = self._tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        if not token_ids:
            raise RuntimeError("chat-template tokenization produced no tokens.")

        return tuple(token_ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a generated-token prefix into cumulative text.

        Args:
            token_ids: Generated token identifiers accumulated so far.

        Returns:
            Decoded text with model special tokens removed.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
