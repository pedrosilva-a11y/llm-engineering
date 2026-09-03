"""Test for language model architecture specifications."""

from dataclasses import replace

import pytest

from serving.cost_models.models import ModelSpecification


@pytest.fixture
def tiny_gpt() -> ModelSpecification:
    """Create a baseline Tiny GPT model specification."""
    return ModelSpecification(
        name="tiny-gpt",
        n_layer=2,
        d_model=64,
        n_head=4,
        n_kv_head=4,
        d_head=16,
        d_ff=256,
        vocab_size=1_000,
    )


def test_model_specification_defaults(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify model fields and default architecture options."""
    assert tiny_gpt.name == "tiny-gpt"
    assert tiny_gpt.n_layer == 2
    assert tiny_gpt.d_model == 64
    assert tiny_gpt.n_head == 4
    assert tiny_gpt.n_kv_head == 4
    assert tiny_gpt.d_head == 16
    assert tiny_gpt.d_ff == 256
    assert tiny_gpt.vocab_size == 1_000
    assert tiny_gpt.gated_mlp is True
    assert tiny_gpt.tied_embeddings is False
    assert tiny_gpt.learned_positional is False
    assert tiny_gpt.max_position is None
    assert tiny_gpt.norm_has_bias is True


# Parameter calculations


def test_default_model_parameter_counts(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify parameter counts for the baseline model specification."""
    assert tiny_gpt.attention_params_per_layer == 16_384
    assert tiny_gpt.mlp_params_per_layer == 49_152
    assert tiny_gpt.norm_params_per_layer == 256
    assert tiny_gpt.params_per_block == 65_792
    assert tiny_gpt.embedding_params == 64_000
    assert tiny_gpt.output_head_params == 64_000
    assert tiny_gpt.transformer_block_params == 131_584
    assert tiny_gpt.total_params == 259_712


def test_classic_mlp_uses_two_projection_matrices(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify a classic MLP uses two projection matrices."""
    model = replace(
        tiny_gpt,
        gated_mlp=False,
    )
    assert model.mlp_params_per_layer == 32_768
    assert model.mlp_params_per_layer < tiny_gpt.mlp_params_per_layer


def test_tied_embeddings_remove_output_head_parameters(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify tied embeddings remove separate output-head parameters."""
    model = replace(
        tiny_gpt,
        tied_embeddings=True,
    )

    assert model.output_head_params == 0
    assert model.total_params == tiny_gpt.total_params - tiny_gpt.output_head_params


def test_normalization_without_bias_excludes_beta_parameters(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify bias-free normalization excludes beta parameters."""
    model = replace(
        tiny_gpt,
        norm_has_bias=False,
    )

    assert model.norm_params_per_layer == 128


def test_attention_parameter_count_with_gqa(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify GQA uses separate query and key/value projection dimensions."""
    model = replace(
        tiny_gpt,
        n_head=8,
        n_kv_head=2,
        d_head=8,
    )

    assert model.attention_params_per_layer == 10_240


# Positional embeddings / validation


def test_learned_positional_embeddings_increase_embedding_params(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify learned positional embeddings contribute to parameter count."""
    model = replace(
        tiny_gpt,
        learned_positional=True,
        max_position=2_048,
    )

    assert model.learned_positional is True
    assert model.max_position == 2_048
    assert model.embedding_params == 195_072


def test_learned_positional_embeddings_require_max_position(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify learned positional embeddings require max_position."""
    with pytest.raises(
        ValueError,
        match="max_position must be set",
    ):
        replace(
            tiny_gpt,
            learned_positional=True,
            max_position=None,
        )


# Memory


def test_default_model_memory_estimates(
    tiny_gpt: ModelSpecification,
) -> None:
    """Verify default KV-cache and model-weight memory estimates."""
    kv_cache_memory = tiny_gpt.kv_bytes_per_token()
    assert kv_cache_memory == 512

    weights_memory = tiny_gpt.weight_bytes()
    assert weights_memory == 519_424


@pytest.mark.parametrize(
    ("bytes_per_value", "expected"),
    [
        (1, 256),
        (2, 512),
        (4, 1_024),
    ],
)
def test_kv_bytes_per_token_scales_with_value_size(
    tiny_gpt: ModelSpecification,
    bytes_per_value: int,
    expected: int,
) -> None:
    """Verify KV-cache memory scales linearly with element size."""
    result = tiny_gpt.kv_bytes_per_token(
        bytes_per_value=bytes_per_value,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("bytes_per_value", "expected"),
    [
        (1, 259_712),
        (2, 519_424),
        (4, 1_038_848),
    ],
)
def test_weight_bytes_scale_with_value_size(
    tiny_gpt: ModelSpecification,
    bytes_per_value: int,
    expected: int,
) -> None:
    """Verify model-weight memory scales linearly with parameter size."""
    result = tiny_gpt.weight_bytes(
        bytes_per_value=bytes_per_value,
    )

    assert result == expected
