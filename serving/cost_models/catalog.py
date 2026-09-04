"""Reusable model and accelerator specifications for cost-model analysis."""

from serving.cost_models.hardware import HardwareSpecification
from serving.cost_models.models import ModelSpecification

LLAMA_3_8B = ModelSpecification(
    name="llama-3-8b",
    n_layer=32,
    d_model=4_096,
    n_head=32,
    n_kv_head=8,
    d_head=128,
    d_ff=14_336,
    vocab_size=128_256,
    norm_has_bias=False,
)


QWEN_2_5_1_5B = ModelSpecification(
    name="qwen2.5-1.5b",
    n_layer=28,
    d_model=1_536,
    n_head=12,
    n_kv_head=2,
    d_head=128,
    d_ff=8_960,
    vocab_size=151_936,
    tied_embeddings=True,
    norm_has_bias=False,
)


# Small decoder-only model used for CPU engine development and integration tests.
TOY_DECODER_MODEL = ModelSpecification(
    name="toy-decoder-model",
    n_layer=2,
    d_model=64,
    n_head=4,
    n_kv_head=2,
    d_head=16,
    d_ff=256,
    vocab_size=256,
    norm_has_bias=False,
)


L4 = HardwareSpecification(
    name="L4",
    peak_bf16_tflops=121.0,
    memory_bandwidth_tb_s=0.3,
    memory_capacity_gib=22.5,
)


H100_SXM = HardwareSpecification(
    name="H100-SXM",
    peak_bf16_tflops=989.5,
    memory_bandwidth_tb_s=3.35,
    memory_capacity_gib=79.6,
)


# Registries


MODEL_CATALOG = {
    LLAMA_3_8B.name: LLAMA_3_8B,
    QWEN_2_5_1_5B.name: QWEN_2_5_1_5B,
    TOY_DECODER_MODEL.name: TOY_DECODER_MODEL,
}

HARDWARE_CATALOG = {
    L4.name: L4,
    H100_SXM.name: H100_SXM,
}
