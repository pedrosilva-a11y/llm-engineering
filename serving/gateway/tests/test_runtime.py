"""Tests for gateway runtime composition."""

import json
from dataclasses import dataclass

import httpx2
import pytest
import torch

import serving.gateway.runtime as runtime
from serving.engine.config import EngineConfiguration
from serving.engine.engine import Engine
from serving.engine.model_runner import (
    DeterministicStubModelRunner,
    ModelRunner,
)
from serving.engine.paged_model_runner import PagedModelRunner
from serving.gateway.tests.helpers import FakeTextTokenizer
from serving.gateway.tokenizer import QWEN_MODEL_NAME, QWEN_MODEL_REVISION


@pytest.fixture
def anyio_backend() -> str:
    """Run async runtime tests with asyncio."""
    return "asyncio"


@dataclass
class _FakeModelConfig:
    """Minimal Qwen-like model configuration used by runtime tests."""

    num_hidden_layers: int = 28
    num_key_value_heads: int = 2
    num_attention_heads: int = 12
    hidden_size: int = 1_536
    head_dim: int = 128
    eos_token_id: int = 151_645


class _FakeCausalModel:
    """Minimal causal model used to test GPU runtime composition."""

    def __init__(self) -> None:
        """Initialize fake model state."""
        self.config = _FakeModelConfig()
        self.dtype = torch.bfloat16
        self.training = True
        self.moved_to: torch.device | None = None

    def to(self, device: torch.device) -> "_FakeCausalModel":
        """Record the target inference device."""
        self.moved_to = device
        return self

    def eval(self) -> "_FakeCausalModel":
        """Record evaluation mode."""
        self.training = False
        return self


class _FakePagedKVStorage:
    """Record paged KV-storage construction without allocating CUDA tensors."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Record supplied storage configuration."""
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device


class _FakeEngine:
    """Record engine composition without initializing CUDA runtime state."""

    def __init__(
        self,
        configuration: EngineConfiguration,
        model_runner: ModelRunner,
    ) -> None:
        """Record the supplied engine dependencies."""
        self.configuration = configuration
        self.model_runner = model_runner


def test_create_development_engine_uses_cpu_stub_runner() -> None:
    """Build the local runtime with deterministic CPU runner."""
    engine = runtime.create_development_engine()

    assert engine.configuration.device == "cpu"
    assert isinstance(engine.model_runner, DeterministicStubModelRunner)


@pytest.mark.anyio
async def test_development_app_stream_completions() -> None:
    """Serve a completion through the fully composed local runtime."""
    tokenizer = FakeTextTokenizer()

    application = runtime.create_development_app(tokenizer=tokenizer)

    async with application.router.lifespan_context(application):
        transport = httpx2.ASGITransport(app=application)

        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "stub-model",
                    "prompt": "Hello",
                    "max_tokens": 2,
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert tokenizer.encoded_prompts == ["Hello"]

    events = _sse_data_events(response)

    assert events[-1] == "[DONE]"

    payloads = [json.loads(event) for event in events[:-1]]

    assert [payload["choices"][0]["token_id"] for payload in payloads] == [
        1,
        1,
    ]
    assert [payload["choices"][0]["cumulative_text"] for payload in payloads] == [
        "1",
        "11",
    ]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"


def test_create_gpu_engine_rejects_missing_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject GPU runtime construction when CUDA is unavailable."""
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA is required",
    ):
        runtime.create_gpu_engine()


def test_create_gpu_engine_rejects_missing_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject GPU runtime construction without BF16 support."""
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="must support BF16",
    ):
        runtime.create_gpu_engine()


def test_create_gpu_engine_uses_paged_qwen_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the CUDA runtime with pinned Qwen and paged KV storage."""
    fake_model = _FakeCausalModel()
    storage_instances: list[_FakePagedKVStorage] = []
    model_load_calls: list[tuple[str, dict[str, object]]] = []

    class _FakeAutoModelForCausalLM:
        """Record model-loading requests."""

        @staticmethod
        def from_pretrained(
            model_name: str,
            **kwargs: object,
        ) -> _FakeCausalModel:
            model_load_calls.append((model_name, dict(kwargs)))
            return fake_model

    class _RecordingPagedKVStorage(_FakePagedKVStorage):
        """Record each constructed KV-storage instance."""

        def __init__(
            self,
            num_layers: int,
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_dim: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            super().__init__(
                num_layers=num_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            storage_instances.append(self)

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime,
        "AutoModelForCausalLM",
        _FakeAutoModelForCausalLM,
    )
    monkeypatch.setattr(
        runtime,
        "PagedKVStorage",
        _RecordingPagedKVStorage,
    )
    monkeypatch.setattr(
        runtime,
        "Engine",
        _FakeEngine,
    )

    engine = runtime.create_gpu_engine()

    assert engine.configuration.device == "cuda"
    assert engine.configuration.eos_token_id == fake_model.config.eos_token_id

    assert isinstance(engine.model_runner, PagedModelRunner)

    assert fake_model.moved_to == torch.device("cuda")
    assert not fake_model.training

    assert len(model_load_calls) == 1

    model_name, model_load_kwargs = model_load_calls[0]

    assert model_name == QWEN_MODEL_NAME
    assert model_load_kwargs["revision"] == QWEN_MODEL_REVISION
    assert model_load_kwargs["dtype"] == torch.bfloat16
    assert model_load_kwargs["trust_remote_code"] is False

    assert len(storage_instances) == 1

    storage = storage_instances[0]

    assert storage.num_layers == fake_model.config.num_hidden_layers
    assert storage.num_blocks == engine.configuration.num_blocks
    assert storage.block_size == engine.configuration.block_size
    assert storage.num_kv_heads == fake_model.config.num_key_value_heads
    assert storage.head_dim == fake_model.config.head_dim
    assert storage.dtype == torch.bfloat16
    assert storage.device == torch.device("cuda")


@pytest.mark.anyio
async def test_gpu_app_uses_injected_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the GPU application with an injected tokenizer dependency."""
    tokenizer = FakeTextTokenizer()
    engine = runtime.create_development_engine()

    def create_test_gpu_engine() -> Engine:
        return engine

    monkeypatch.setattr(
        runtime,
        "create_gpu_engine",
        create_test_gpu_engine,
    )

    application = runtime.create_gpu_app(tokenizer=tokenizer)

    async with application.router.lifespan_context(application):
        transport = httpx2.ASGITransport(app=application)

        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": QWEN_MODEL_NAME,
                    "prompt": "Hello GPU runtime",
                    "max_tokens": 2,
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert tokenizer.encoded_prompts == ["Hello GPU runtime"]

    events = _sse_data_events(response)

    assert events[-1] == "[DONE]"

    payloads = [json.loads(event) for event in events[:-1]]

    assert [payload["choices"][0]["token_id"] for payload in payloads] == [
        1,
        1,
    ]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"


def _sse_data_events(response: httpx2.Response) -> list[str]:
    """Extract SSE data payloads from an HTTP response."""
    return [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
