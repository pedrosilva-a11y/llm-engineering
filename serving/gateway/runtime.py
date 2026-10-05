"""Runtime composition for the inference HTTP gateway."""

from typing import cast

import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, PreTrainedModel

from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import DeterministicStubModelRunner
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.paged_model_runner import PagedModelRunner
from serving.gateway.app import create_gateway_app
from serving.gateway.tokenizer import (
    QWEN_MODEL_NAME,
    QWEN_MODEL_REVISION,
    QwenTextTokenizer,
    TextTokenizer,
)


def create_development_engine() -> Engine:
    """Create the deterministic CPU engine used for local gateway development."""
    configuration = EngineConfiguration()

    model_runner = DeterministicStubModelRunner(
        vocab_size=32,
        eos_token_id=configuration.eos_token_id,
        generated_token_id=1,
        default_eos_after=1_000_000,
        device=configuration.device,
    )

    return Engine(
        configuration=configuration,
        model_runner=model_runner,
    )


def create_development_app(*, tokenizer: TextTokenizer | None = None) -> FastAPI:
    """Create the local development gateway application.

    The development application uses the deterministic CPU inference engine and
    accepts an optional tokenizer dependency. When no tokenizer is supplied, the
    pinned Qwen tokenizer is loaded so the local gateway exposes the same
    text-facing request and response behavior expected by the final demo runtime.

    Supplying a tokenizer is useful for tests, where a deterministic fake can be
    injected to avoid network access and Hugging Face dependency loading.

    Args:
        tokenizer: Optional text tokenizer used to encode incoming prompts and
            decode generated token sequences. If omitted, the pinned
            ``QwenTextTokenizer`` is loaded from pretrained artifacts.

    Returns:
        FastAPI application configured with the development inference engine and
        the selected tokenizer.
    """
    gateway_tokenizer = (
        tokenizer if tokenizer is not None else QwenTextTokenizer.from_pretrained()
    )

    return create_gateway_app(
        engine=create_development_engine(),
        tokenizer=gateway_tokenizer,
    )


def create_gpu_engine() -> Engine:
    """Create the CUDA inference engine backed by the pinned Qwen model.

    Returns:
        Inference engine configured with a BF16 Qwen model and paged KV cache.

    Raises:
        RuntimeError: If CUDA or BF16 execution is unavailable, or required
            model configuration values cannot be resolved.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU inference runtime.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured CUDA device must support BF16.")

    device = torch.device("cuda")

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            revision=QWEN_MODEL_REVISION,
            dtype=torch.bfloat16,
            trust_remote_code=False,
        ),
    )

    cast(torch.nn.Module, model).to(device)

    eos_token_id = _model_eos_token_id(model)

    configuration = EngineConfiguration(
        device=str(device),
        eos_token_id=eos_token_id,
    )

    storage = PagedKVStorage(
        num_layers=_positive_model_config_int(model, "num_hidden_layers"),
        num_blocks=configuration.num_blocks,
        block_size=configuration.block_size,
        num_kv_heads=_positive_model_config_int(model, "num_key_value_heads"),
        head_dim=_model_head_dim(model),
        dtype=model.dtype,
        device=device,
    )

    paged_cache = PagedKVCache(storage)

    model_runner = PagedModelRunner(
        model=model,
        device=device,
        paged_cache=paged_cache,
    )

    return Engine(
        configuration=configuration,
        model_runner=model_runner,
    )


def create_gpu_app(*, tokenizer: TextTokenizer | None = None) -> FastAPI:
    """Create the GPU-backed inference gateway application.

    Args:
        tokenizer: Optional tokenizer dependency. If omitted, the pinned Qwen
            tokenizer is loaded from pretrained artifacts.

    Returns:
        FastAPI application backed by the CUDA Qwen inference engine.
    """
    gateway_tokenizer = (
        tokenizer if tokenizer is not None else QwenTextTokenizer.from_pretrained()
    )

    return create_gateway_app(
        engine=create_gpu_engine(),
        tokenizer=gateway_tokenizer,
    )


def _positive_model_config_int(
    model: PreTrainedModel,
    attribute: str,
) -> int:
    """Read one required positive integer from the model configuration.

    Args:
        model: Loaded Hugging Face model.
        attribute: Configuration attribute to read.

    Returns:
        Positive integer stored in the model configuration.

    Raises:
        RuntimeError: If the configuration value is missing or invalid.
    """
    value = cast(object, getattr(model.config, attribute, None))

    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"Model configuration attribute {attribute!r} must be a positive integer.",
        )

    return value


def _model_eos_token_id(model: PreTrainedModel) -> int:
    """Return the model end-of-sequence token identifier.

    Args:
        model: Loaded Hugging Face model.

    Returns:
        Non-negative EOS token identifier.

    Raises:
        RuntimeError: If the model configuration does not expose a valid EOS ID.
    """
    value = cast(object, getattr(model.config, "eos_token_id", None))

    if not isinstance(value, int) or value < 0:
        raise RuntimeError(
            "Model configuration must expose a non-negative eos_token_id.",
        )

    return value


def _model_head_dim(model: PreTrainedModel) -> int:
    """Return the attention head dimension from the model configuration.

    Args:
        model: Loaded Hugging Face model.

    Returns:
        Dimension of one attention head.

    Raises:
        RuntimeError: If the head dimension cannot be derived consistently.
    """
    configured_head_dim = cast(
        object,
        getattr(model.config, "head_dim", None),
    )

    if isinstance(configured_head_dim, int) and configured_head_dim > 0:
        return configured_head_dim

    hidden_size = _positive_model_config_int(model, "hidden_size")
    num_attention_heads = _positive_model_config_int(
        model,
        "num_attention_heads",
    )

    if hidden_size % num_attention_heads != 0:
        raise RuntimeError(
            "Model hidden_size must be divisible by num_attention_heads.",
        )

    return hidden_size // num_attention_heads
