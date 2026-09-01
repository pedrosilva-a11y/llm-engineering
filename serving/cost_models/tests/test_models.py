"""Test for language model architecture specifications."""

from serving.cost_models.models import ModelSpecification


def test_model_specification_success() -> None:
    """Verify valid model specification values are stored correctly."""
    model = ModelSpecification(
        name="tiny-gpt",
        n_layer=2,
        d_model=64,
        n_head=4,
        n_kv_head=4,
        d_head=16,
        d_ff=256,
        vocab_size=1_000,
    )

    assert model.name == "tiny-gpt"
    assert model.n_layer == 2
    assert model.d_model == 64
    assert model.n_head == 4
    assert model.n_kv_head == 4
    assert model.d_head == 16
    assert model.d_ff == 256
    assert model.vocab_size == 1_000
