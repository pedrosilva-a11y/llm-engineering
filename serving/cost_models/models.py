"""Data structures for language model architecture specifications."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpecification:
    """Specification for decoder-only language model architecture.

    Parameter-counting assumptions:
    The current implementation counts attention projection weights
    W_Q, W_K, W_V, and W_O, together with the MLP
    projection matrices. Classic feed-forward blocks use two MLP
    matrices, while gated architectures such as SwiGLU use three.

    Attention and MLP projection biases are currently excluded. This
    matches bias-free architectures such as Llama, but not every
    decoder-only Transformer. For example, Qwen2.5 uses QKV attention
    biases, so this implementation cannot produce an exact Qwen2.5
    parameter count yet.

    Bias support should be added only when required by a target
    architecture. This keeps ModelSpecification minimal while making
    its current parameter-counting scope explicit.

    Attributes:
        name: Human-readable model identifier.
        n_layer: Number of transformer blocks.
        d_model: Hidden/embedding dimension.
        n_head: Number of query attention heads.
        n_kv_head: Number of key/value heads.
        d_head: Dimension of each attention head.
        d_ff: Feed-forward intermediate dimension.
        vocab_size: Tokenizer vocabulary size.
        gated_mlp: Whether the feed-forward network uses a gated
            multi-layer perceptron architecture such as SwiGLU.
        tied_embeddings: Whether the token embedding matrix is shared with
            the output language-model head.
        learned_positional: Whether the model uses learned positional
            embeddings instead of a non-parametric method such as RoPE.
        max_position: Maximum sequence position supported by the learned
            positional embeddings table. Required when learned_positional
            is True.
        norm_has_bias: Whether normalization layers include a learnable
            bias parameter in addition to scale.
    """

    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_head: int
    d_ff: int
    vocab_size: int
    gated_mlp: bool = True
    tied_embeddings: bool = False
    learned_positional: bool = False
    max_position: int | None = None
    norm_has_bias: bool = True

    def __post_init__(self) -> None:
        """Validate model specification consistency.

        Raises:
            ValueError:
                If max_position is None when learned_positional is True.
        """
        if self.learned_positional and self.max_position is None:
            raise ValueError(
                "max_position must be set when learned_positional is True.",
            )

    @property
    def attention_params_per_layer(self) -> int:
        """Calculate attention projection parameters per transformer layer.

        Query and output projections use n_head * d_head, while key and value
        projections use n_kv_head * d_head. These dimensions differ under
        Grouped Query Attention.

        Returns:
            Attention parameters count per layer.
        """
        q_dim = self.n_head * self.d_head
        kv_dim = self.n_kv_head * self.d_head

        return (
            2 * (self.d_model * q_dim)  # W_Q and W_O
            + 2 * (self.d_model * kv_dim)  # W_K and W_V
        )

    @property
    def mlp_params_per_layer(self) -> int:
        """Calculate MLP parameters per transformer layer.

        A gated MLP uses three projection matrices, while a classic
        feed-forward network uses two.

        Returns:
            Number of MLP parameters per layer.
        """
        matrices = 3 if self.gated_mlp else 2
        return matrices * self.d_model * self.d_ff

    @property
    def norm_params_per_layer(self) -> int:
        """Calculate normalization parameters per transformer layer.

        Two normalization steps per block, each normalization with gamma
        and optionally beta.

        Returns:
            Number of normalization parameters per layer.
        """
        per_norm = self.d_model * (2 if self.norm_has_bias else 1)
        return 2 * per_norm

    @property
    def params_per_block(self) -> int:
        """Calculate total parameters in one transformer block.

        Includes attention projections, MLP projections, and normalization
        parameters.

        Returns:
            Number of parameters per transformer block.
        """
        return (
            self.attention_params_per_layer
            + self.mlp_params_per_layer
            + self.norm_params_per_layer
        )

    @property
    def embedding_params(self) -> int:
        """Calculate embedding parameters.

        Includes token embeddings and, when enabled, learned positional
        embeddings.

        Returns:
            Total number of embedding parameters.
        """
        total = self.vocab_size * self.d_model
        if self.learned_positional and self.max_position is not None:
            total += self.max_position * self.d_model
        return total

    @property
    def output_head_params(self) -> int:
        """Calculate output language-model head parameters.

        Returns zero when token embeddings are tied with the output head.

        Returns:
            Number of output head parameters.
        """
        return 0 if self.tied_embeddings else self.vocab_size * self.d_model

    @property
    def transformer_block_params(self) -> int:
        """Calculate parameters across all transformer blocks.

        This value excludes embeddings, the final normalization layer, and the
        output head. It is also useful for approximate per-token FLOP estimates.

        Returns:
            Total number of parameters across all transformer blocks.
        """
        return self.n_layer * self.params_per_block

    @property
    def total_params(self) -> int:
        """Calculate the total number of model parameters.

        Includes token and positional embeddings, all transformer-block
        parameters, the final normalization layer, and the output head
        when embeddings are not tied.

        Returns:
            Total number of model parameters.
        """
        final_norm = self.d_model * (2 if self.norm_has_bias else 1)
        return (
            self.embedding_params
            + self.transformer_block_params
            + final_norm
            + self.output_head_params
        )

    def kv_bytes_per_token(
        self,
        bytes_per_value: int = 2,
    ) -> int:
        """Calculate KV-cache memory required per token.

        Accounts for both key and value tensors across all transformer
        layers, using the configured number of KV heads and head dimension.

        Args:
            bytes_per_value: Number of bytes used to store each KV element.

        Returns:
            KV-cache memory required per token, in bytes.
        """
        return 2 * self.n_layer * self.n_kv_head * self.d_head * bytes_per_value

    def weight_bytes(
        self,
        bytes_per_value: int = 2,
    ) -> int:
        """Calculate memory required to store all model weights.

        Args:
            bytes_per_value: Number of bytes used to store each model
                parameter.

        Returns:
            Total model-weight memory, in bytes.
        """
        return self.total_params * bytes_per_value
