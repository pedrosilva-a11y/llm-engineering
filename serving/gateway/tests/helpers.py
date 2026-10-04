"""Shared test helpers for the inference HTTP gateway."""

from collections.abc import Sequence


class FakeTextTokenizer:
    """Deterministic text tokenizer used by gateway tests.

    The fake records prompt-encoding and token-decoding calls so tests can verify
    how the gateway interacts with the tokenizer without loading Hugging Face
    artifacts or requiring network access.

    Encoded prompt token IDs are configurable at construction time. Decoding is
    deterministic and returns the concatenated decimal representation of the
    supplied token IDs, which makes cumulative streaming behavior easy to assert.
    """

    def __init__(
        self,
        *,
        encoded_token_ids: tuple[int, ...] = (1, 2, 3),
        encode_error: ValueError | None = None,
    ) -> None:
        """Initialize deterministic tokenizer behavior.

        Args:
            encoded_token_ids: Token IDs returned whenever ``encode_prompt`` is
                called.
            encode_error: Optional error raised when ``encode_prompt`` is called.
        """
        self.encoded_token_ids = encoded_token_ids
        self.encode_error = encode_error
        self.encoded_prompts: list[str] = []
        self.decoded_token_ids: list[tuple[int, ...]] = []

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        """Record and deterministically encode one text prompt.

        Args:
            prompt: Human-readable prompt submitted by the gateway.

        Returns:
            Configured token IDs representing the encoded prompt.

        Raises:
            ValueError: If the fake is configured to reject prompt encoding.
        """
        self.encoded_prompts.append(prompt)

        if self.encode_error is not None:
            raise self.encode_error

        return self.encoded_token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Record and deterministically decode generated token IDs.

        Args:
            token_ids: Generated token identifiers accumulated so far.

        Returns:
            Concatenated decimal representation of the supplied token IDs.
        """
        token_tuple = tuple(token_ids)
        self.decoded_token_ids.append(token_tuple)

        return "".join(str(token_id) for token_id in token_tuple)
