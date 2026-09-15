"""Tests for the paged KV-cache model runner."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import PreTrainedModel

from serving.engine.execution import ExecutionPhase, ModelExecution
from serving.engine.hf_paged_cache import HFPagedCache
from serving.engine.paged_kv_cache import PagedKVCache
from serving.engine.paged_kv_storage import PagedKVStorage
from serving.engine.paged_model_runner import PagedModelRunner
from serving.engine.request import Request
from serving.engine.sequence import SequenceState

VOCAB_SIZE = 32
NUM_LAYERS = 1
NUM_BLOCKS = 4
BLOCK_SIZE = 8
NUM_KV_HEADS = 2
HEAD_DIM = 4
DTYPE = torch.float32
DEVICE = torch.device("cpu")


class RecordingCausalModel(torch.nn.Module):
    """Record model calls while exercising the Hugging Face cache adapter."""

    def __init__(self) -> None:
        """Initialize the recording model."""
        super().__init__()
        self.input_id_calls: list[tuple[int, ...]] = []
        self.cache_calls: list[HFPagedCache] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: HFPagedCache,
        use_cache: bool,
    ) -> SimpleNamespace:
        """Return deterministic logits and write synthetic KV states."""
        if not use_cache:
            raise AssertionError("Paged model execution must enable caching.")

        self.input_id_calls.append(
            tuple(int(token_id) for token_id in input_ids[0].tolist())
        )
        self.cache_calls.append(past_key_values)

        query_length = input_ids.shape[1]

        keys = (
            input_ids.to(dtype=DTYPE)
            .reshape(1, 1, query_length, 1)
            .expand(
                1,
                NUM_KV_HEADS,
                query_length,
                HEAD_DIM,
            )
            .clone()
        )
        values = keys + 1000.0

        past_key_values.update(
            key_states=keys,
            value_states=values,
            layer_idx=0,
        )

        logits = torch.full(
            (1, query_length, VOCAB_SIZE),
            fill_value=float("-inf"),
            dtype=DTYPE,
            device=input_ids.device,
        )

        next_token_id = int(input_ids[0, -1]) % VOCAB_SIZE
        logits[0, -1, next_token_id] = 0.0

        return SimpleNamespace(logits=logits)


@pytest.fixture
def storage() -> PagedKVStorage:
    """Create small CPU paged KV storage."""
    return PagedKVStorage(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )


@pytest.fixture
def recording_model() -> RecordingCausalModel:
    """Create a model that records paged runner invocations."""
    return RecordingCausalModel()


def make_sequence(
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: tuple[int, ...] = (),
    request_id: str = "request",
) -> SequenceState:
    """Create a sequence with optional generated token history."""
    sequence = SequenceState(
        request=Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=8,
        )
    )

    for token_id in generated_token_ids:
        sequence.append_token(token_id)

    return sequence


def make_execution(
    sequence: SequenceState,
    phase: ExecutionPhase,
    slot_ids: tuple[int, ...] | None = None,
) -> ModelExecution:
    """Create model execution metadata for a sequence."""
    if slot_ids is None:
        slot_ids = tuple(range(sequence.current_length))

    return ModelExecution(
        sequence=sequence,
        phase=phase,
        slot_ids=slot_ids,
    )


def make_runner(
    model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> PagedModelRunner:
    """Create a paged model runner backed by CPU storage."""
    return PagedModelRunner(
        model=cast(PreTrainedModel, model),
        device=DEVICE,
        paged_cache=PagedKVCache(storage),
    )


def test_prefill_uses_complete_sequence_history(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Send the complete logical sequence to the model during prefill."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(prompt_token_ids=(1, 4, 5))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11),
    )

    runner.forward((execution,))

    assert recording_model.input_id_calls == [(1, 4, 5)]


def test_decode_uses_only_newest_token(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Send only the newest logical token to the model during decode."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(
        prompt_token_ids=(1, 4, 5),
        generated_token_ids=(6,),
    )

    previous_keys = torch.zeros(
        (3, NUM_KV_HEADS, HEAD_DIM),
        dtype=DTYPE,
    )
    previous_values = torch.zeros_like(previous_keys)

    storage.scatter(
        layer_index=0,
        slot_ids=(7, 2, 11),
        keys=previous_keys,
        values=previous_values,
    )

    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(7, 2, 11, 5),
    )

    runner.forward((execution,))

    assert recording_model.input_id_calls == [(6,)]


def test_reprefill_uses_complete_generated_history(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Send complete prompt and generated history during re-prefill."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(
        prompt_token_ids=(1, 4, 5),
        generated_token_ids=(6, 7),
    )
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11, 5, 14),
    )

    runner.forward((execution,))

    assert recording_model.input_id_calls == [(1, 4, 5, 6, 7)]


def test_forward_returns_one_logit_vector_per_execution(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Return one next-token logit vector for each execution."""
    runner = make_runner(recording_model, storage)

    first = make_execution(
        sequence=make_sequence(
            prompt_token_ids=(1, 4),
            request_id="first",
        ),
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2),
    )
    second = make_execution(
        sequence=make_sequence(
            prompt_token_ids=(1, 9, 10),
            request_id="second",
        ),
        phase=ExecutionPhase.PREFILL,
        slot_ids=(8, 9, 10),
    )

    logits = runner.forward((first, second))

    assert logits.shape == (2, VOCAB_SIZE)
    assert recording_model.input_id_calls == [
        (1, 4),
        (1, 9, 10),
    ]


def test_forward_preserves_execution_order(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Stack output logits in the same order as model executions."""
    runner = make_runner(recording_model, storage)

    first = make_execution(
        sequence=make_sequence(
            prompt_token_ids=(1, 4),
            request_id="first",
        ),
        phase=ExecutionPhase.PREFILL,
        slot_ids=(1, 2),
    )
    second = make_execution(
        sequence=make_sequence(
            prompt_token_ids=(1, 9),
            request_id="second",
        ),
        phase=ExecutionPhase.PREFILL,
        slot_ids=(8, 9),
    )

    logits = runner.forward((first, second))

    assert int(torch.argmax(logits[0])) == 4
    assert int(torch.argmax(logits[1])) == 9


def test_forward_passes_hf_paged_cache_to_model(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Provide a Hugging Face paged cache for each model execution."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(prompt_token_ids=(1, 4, 5))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11),
    )

    runner.forward((execution,))

    assert len(recording_model.cache_calls) == 1
    assert isinstance(recording_model.cache_calls[0], HFPagedCache)


def test_prefill_writes_kv_states_to_physical_slots(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Write model-produced prefill states into assigned physical slots."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(prompt_token_ids=(1, 4, 5))
    execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11),
    )

    runner.forward((execution,))

    stored_keys, stored_values = storage.gather(
        layer_index=0,
        slot_ids=execution.slot_ids,
    )

    expected_tokens = torch.tensor(
        [1.0, 4.0, 5.0],
        dtype=DTYPE,
    ).reshape(3, 1, 1)

    expected_keys = expected_tokens.expand(
        3,
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    assert torch.equal(stored_keys, expected_keys)
    assert torch.equal(stored_values, expected_keys + 1000.0)


def test_decode_writes_only_newest_physical_slot(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Preserve existing KV history while writing the newest decode state."""
    runner = make_runner(recording_model, storage)

    sequence = make_sequence(prompt_token_ids=(1, 4, 5))
    prefill_execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.PREFILL,
        slot_ids=(7, 2, 11),
    )

    runner.forward((prefill_execution,))

    sequence.append_token(6)

    decode_execution = make_execution(
        sequence=sequence,
        phase=ExecutionPhase.DECODE,
        slot_ids=(7, 2, 11, 5),
    )

    runner.forward((decode_execution,))

    stored_keys, stored_values = storage.gather(
        layer_index=0,
        slot_ids=decode_execution.slot_ids,
    )

    expected_tokens = torch.tensor(
        [1.0, 4.0, 5.0, 6.0],
        dtype=DTYPE,
    ).reshape(4, 1, 1)

    expected_keys = expected_tokens.expand(
        4,
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    assert torch.equal(stored_keys, expected_keys)
    assert torch.equal(stored_values, expected_keys + 1000.0)


def test_forward_returns_logits_on_configured_device(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Return next-token logits on the configured inference device."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(prompt_token_ids=(1, 4, 5))

    logits = runner.forward(
        (
            make_execution(
                sequence=sequence,
                phase=ExecutionPhase.PREFILL,
            ),
        )
    )

    assert logits.device.type == "cpu"


def test_constructor_sets_model_to_evaluation_mode(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Put the underlying causal language model into evaluation mode."""
    recording_model.train()

    make_runner(recording_model, storage)

    assert not recording_model.training


def test_forward_rejects_empty_batch(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Reject inference without any model executions."""
    runner = make_runner(recording_model, storage)

    with pytest.raises(
        ValueError,
        match="At least one execution is required.",
    ):
        runner.forward(())


def test_forward_rejects_unsupported_execution_phase(
    recording_model: RecordingCausalModel,
    storage: PagedKVStorage,
) -> None:
    """Reject model executions with unsupported execution phases."""
    runner = make_runner(recording_model, storage)
    sequence = make_sequence(prompt_token_ids=(1,))

    execution = ModelExecution(
        sequence=sequence,
        phase=cast(ExecutionPhase, "unsupported"),
        slot_ids=(1,),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported execution phase",
    ):
        runner.forward((execution,))
