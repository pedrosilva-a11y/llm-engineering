"""Data structures for language model architecture specifications."""

from dataclasses import dataclass


@dataclass
class ModelSpecification:
    """Specification for decoder-only language model architecture.

    Attributes:
        name: Human-readable model identifier.
        n_layer: Number of transformer blocks.
        d_model: Hidden/embedding dimension.
        n_head: Number of query attention heads.
        n_kv_head: Number of key/value heads.
        d_head: Dimension of each attention head.
        d_ff: Feed-forward intermediate dimension.
        vocab_size: Tokenizer vocabulary size.
    """

    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_head: int
    d_ff: int
    vocab_size: int
