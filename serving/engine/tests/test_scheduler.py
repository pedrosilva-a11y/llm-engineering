"""Tests for LLM inference engine scheduling."""

import pytest

from serving.engine.config import EngineConfiguration
from serving.engine.request import Request
from serving.engine.scheduler import Scheduler
from serving.engine.sequence import FinishReason, SequenceStatus

# Fixtures


@pytest.fixture
def configuration() -> EngineConfiguration:
    """Create scheduler configuration for control-flow tests."""
    return EngineConfiguration(
        block_size=16,
        num_blocks=4,
        max_sequences=2,
        max_batched_tokens=64,
    )


@pytest.fixture
def scheduler(
    configuration: EngineConfiguration,
) -> Scheduler:
    """Create a scheduler with deterministic capacity limits."""
    return Scheduler(
        configuration=configuration,
    )


def assert_scheduler_invariant(scheduler: Scheduler) -> None:
    """Assert queue membership agrees with KV-cache ownership."""
    waiting_ids = {
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    }
    running_ids = {
        sequence.request.request_id for sequence in scheduler.running_sequences
    }
    finished_ids = {
        sequence.request.request_id for sequence in scheduler.finished_sequences
    }

    for request_id in waiting_ids | finished_ids:
        with pytest.raises(ValueError, match="Unknown request_id"):
            scheduler.block_table.blocks_for_request(request_id)

    for request_id in running_ids:
        assert scheduler.block_table.blocks_for_request(request_id)

    assert (
        scheduler.block_pool.num_allocated_blocks + scheduler.block_pool.num_free_blocks
        == scheduler.block_pool.num_blocks
    )


# Scheduler initialization


def test_scheduler_initializes_empty(
    scheduler: Scheduler,
) -> None:
    """Initialize scheduler queues and KV-cache state as empty."""
    assert scheduler.waiting_sequences == ()
    assert scheduler.running_sequences == ()
    assert scheduler.finished_sequences == ()
    assert scheduler.has_unfinished_requests is False
    assert scheduler.num_preemptions == 0

    assert scheduler.block_pool.num_blocks == 4
    assert scheduler.block_pool.num_free_blocks == 4
    assert scheduler.block_pool.num_allocated_blocks == 0

    assert scheduler.block_table.block_size == 16


# Submission


def test_submit_adds_request_to_waiting_queue(
    scheduler: Scheduler,
) -> None:
    """Place a submitted request in the waiting queue without admitting it."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=8,
    )

    scheduler.submit(request)

    assert len(scheduler.waiting_sequences) == 1
    assert scheduler.running_sequences == ()
    assert scheduler.finished_sequences == ()

    sequence = scheduler.waiting_sequences[0]

    assert sequence.request is request
    assert sequence.status == SequenceStatus.WAITING
    assert scheduler.has_unfinished_requests is True


def test_submit_does_not_allocate_kv_blocks(
    scheduler: Scheduler,
) -> None:
    """Keep KV-cache blocks free until a waiting request is admitted."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=8,
    )

    scheduler.submit(request)

    assert scheduler.block_pool.num_free_blocks == 4
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_submit_preserves_first_come_first_served_order(
    scheduler: Scheduler,
) -> None:
    """Keep submitted requests in first-come-first-served waiting order."""
    first_request = Request(
        request_id="request-1",
        prompt_token_ids=(1,),
        max_new_tokens=8,
    )
    second_request = Request(
        request_id="request-2",
        prompt_token_ids=(2,),
        max_new_tokens=8,
    )

    scheduler.submit(first_request)
    scheduler.submit(second_request)

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["request-1", "request-2"]


def test_duplicate_request_id_is_rejected(
    scheduler: Scheduler,
) -> None:
    """Reject a request identifier already known to the scheduler."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1, 2),
            max_new_tokens=8,
        ),
    )

    with pytest.raises(ValueError, match="has already been submitted"):
        scheduler.submit(
            Request(
                request_id="request-1",
                prompt_token_ids=(3, 4),
                max_new_tokens=8,
            ),
        )

    assert len(scheduler.waiting_sequences) == 1


def test_prompt_exceeding_token_budget_is_rejected() -> None:
    """Reject prompts that cannot fit in one prefill budget."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_batched_tokens=16,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(17)),
        max_new_tokens=8,
    )

    with pytest.raises(ValueError, match="exceeds max_batched_tokens"):
        scheduler.submit(request)

    assert scheduler.waiting_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_prompt_equal_to_token_budget_is_accepted() -> None:
    """Accept a prompt exactly equal to the per-step token budget."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=16,
        max_batched_tokens=16,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(16)),
        max_new_tokens=1,
    )

    scheduler.submit(request)

    assert len(scheduler.waiting_sequences) == 1


def test_prompt_exceeding_total_kv_capacity_is_rejected() -> None:
    """Reject prompts requiring more blocks than the entire KV-cache pool."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_batched_tokens=64,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(33)),
        max_new_tokens=8,
    )

    with pytest.raises(
        ValueError,
        match="more KV-cache blocks than the scheduler owns",
    ):
        scheduler.submit(request)

    assert scheduler.waiting_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_prompt_equal_to_total_kv_capacity_is_accepted() -> None:
    """Accept a prompt that exactly fills the scheduler's KV-cache capacity."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_batched_tokens=32,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(32)),
        max_new_tokens=1,
    )

    scheduler.submit(request)

    assert len(scheduler.waiting_sequences) == 1
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_submit_rejects_maximum_history_exceeding_reprefill_budget() -> None:
    """Reject a request whose maximum KV-backed history cannot be re-prefilled."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_batched_tokens=16,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(8)),
        max_new_tokens=10,
    )

    with pytest.raises(
        ValueError,
        match="Maximum non-terminal sequence length exceeds max_batched_tokens",
    ):
        scheduler.submit(request)

    assert scheduler.waiting_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_submit_rejects_maximum_history_exceeding_total_kv_capacity() -> None:
    """Reject a request whose maximum KV-backed history exceeds total KV capacity."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=2,
        max_batched_tokens=12,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=(1, 2, 3, 4),
        max_new_tokens=6,
    )

    with pytest.raises(
        ValueError,
        match="more KV-cache blocks than the scheduler owns",
    ):
        scheduler.submit(request)

    assert scheduler.waiting_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_submit_excludes_terminal_token_from_maximum_kv_requirement() -> None:
    """Exclude the terminal generated token from maximum KV-cache requirements."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=1,
        max_batched_tokens=16,
    )
    scheduler = Scheduler(configuration=configuration)

    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(15)),
        max_new_tokens=2,
    )

    scheduler.submit(request)

    assert len(scheduler.waiting_sequences) == 1
    assert scheduler.waiting_sequences[0].request is request
    assert scheduler.block_pool.num_allocated_blocks == 0


# Admission


def test_admit_waiting_sequence_allocates_prompt_blocks(
    scheduler: Scheduler,
) -> None:
    """Allocate prompt blocks and move an admitted sequence to running."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(17)),
        max_new_tokens=8,
    )

    scheduler.submit(request)

    admitted = scheduler._admit_waiting_sequences()

    assert len(admitted) == 1

    sequence = admitted[0]

    assert sequence.request is request
    assert sequence.status == SequenceStatus.RUNNING

    assert scheduler.waiting_sequences == ()
    assert scheduler.running_sequences == (sequence,)

    assert scheduler.block_table.blocks_for_request("request-1") == (0, 1)

    assert scheduler.block_pool.num_allocated_blocks == 2
    assert scheduler.block_pool.num_free_blocks == 2


def test_admission_uses_current_length_for_kv_capacity() -> None:
    """Keep a sequence waiting when its current history cannot fit available KV."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=3,
        max_sequences=2,
        max_batched_tokens=16,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="running-request",
            prompt_token_ids=(1, 2, 3, 4),
            max_new_tokens=1,
        ),
    )
    scheduler._admit_waiting_sequences()

    assert scheduler.block_pool.num_free_blocks == 2

    scheduler.submit(
        Request(
            request_id="waiting-request",
            prompt_token_ids=(5, 6, 7, 8),
            max_new_tokens=6,
        ),
    )

    sequence = scheduler.waiting_sequences[0]
    sequence.generated_token_ids.extend([9, 10, 11, 12, 13])

    admitted = scheduler._admit_waiting_sequences()

    assert sequence.current_length == 9
    assert admitted == ()
    assert scheduler.waiting_sequences == (sequence,)
    assert len(scheduler.running_sequences) == 1
    assert scheduler.block_pool.num_allocated_blocks == 1
    assert scheduler.block_pool.num_free_blocks == 2


def test_admission_allocates_blocks_for_current_sequence_history() -> None:
    """Allocate KV blocks for prompt and previously generated token history."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=4,
        max_sequences=2,
        max_batched_tokens=16,
    )
    scheduler = Scheduler(configuration=configuration)

    request_id = "request-1"

    scheduler.submit(
        Request(
            request_id=request_id,
            prompt_token_ids=(1, 2, 3, 4),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler.waiting_sequences[0]
    sequence.generated_token_ids.extend([5, 6, 7, 8, 9])

    admitted = scheduler._admit_waiting_sequences()

    assert sequence.current_length == 9
    assert admitted == (sequence,)
    assert scheduler.block_table.blocks_for_request(request_id) == (0, 1, 2)
    assert scheduler.block_pool.num_allocated_blocks == 3
    assert scheduler.block_pool.num_free_blocks == 1


def test_admit_waiting_sequences_returns_sequences_in_fcfs_order(
    scheduler: Scheduler,
) -> None:
    """Return admitted sequences in first-come-first-served order."""
    first_request = Request(
        request_id="request-1",
        prompt_token_ids=(1,),
        max_new_tokens=8,
    )
    second_request = Request(
        request_id="request-2",
        prompt_token_ids=(2,),
        max_new_tokens=8,
    )

    scheduler.submit(first_request)
    scheduler.submit(second_request)

    admitted = scheduler._admit_waiting_sequences()

    assert [sequence.request.request_id for sequence in admitted] == [
        "request-1",
        "request-2",
    ]

    assert [
        sequence.request.request_id for sequence in scheduler.running_sequences
    ] == ["request-1", "request-2"]


def test_admission_stops_at_max_sequences() -> None:
    """Stop admission when the configured sequence limit is reached."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_sequences=2,
        max_batched_tokens=64,
    )
    scheduler = Scheduler(configuration=configuration)

    for request_id in ("request-1", "request-2", "request-3"):
        scheduler.submit(
            Request(
                request_id=request_id,
                prompt_token_ids=(1,),
                max_new_tokens=8,
            ),
        )

    admitted = scheduler._admit_waiting_sequences()

    assert len(admitted) == 2
    assert len(scheduler.running_sequences) == 2

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["request-3"]

    assert scheduler.block_pool.num_allocated_blocks == 2
    assert scheduler.block_pool.num_free_blocks == 6


def test_admission_stops_when_kv_blocks_are_insufficient() -> None:
    """Stop admission when the next prompt cannot fit in free KV-cache blocks."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=3,
        max_sequences=4,
        max_batched_tokens=64,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=tuple(range(32)),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=tuple(range(32)),
            max_new_tokens=8,
        ),
    )

    admitted = scheduler._admit_waiting_sequences()

    assert [sequence.request.request_id for sequence in admitted] == ["request-1"]

    assert [
        sequence.request.request_id for sequence in scheduler.running_sequences
    ] == ["request-1"]

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["request-2"]

    assert scheduler.block_pool.num_allocated_blocks == 2
    assert scheduler.block_pool.num_free_blocks == 1


def test_admission_preserves_head_of_line_blocking() -> None:
    """Do not skip an older request that cannot fit in available KV memory."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=3,
        max_sequences=3,
        max_batched_tokens=64,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="running-request",
            prompt_token_ids=tuple(range(32)),
            max_new_tokens=8,
        ),
    )

    scheduler._admit_waiting_sequences()

    assert scheduler.block_pool.num_free_blocks == 1

    scheduler.submit(
        Request(
            request_id="large-request",
            prompt_token_ids=tuple(range(32)),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="small-request",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    admitted = scheduler._admit_waiting_sequences()

    assert admitted == ()

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == [
        "large-request",
        "small-request",
    ]

    assert [
        sequence.request.request_id for sequence in scheduler.running_sequences
    ] == ["running-request"]

    assert scheduler.block_pool.num_free_blocks == 1


def test_waiting_sequence_does_not_receive_kv_blocks_when_admission_stops() -> None:
    """Keep KV blocks unallocated for requests that remain waiting."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=1,
        max_sequences=2,
        max_batched_tokens=32,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(2,),
            max_new_tokens=8,
        ),
    )

    scheduler._admit_waiting_sequences()

    assert scheduler.block_table.blocks_for_request("request-1") == (0,)

    with pytest.raises(ValueError, match="Unknown request_id"):
        scheduler.block_table.blocks_for_request("request-2")

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["request-2"]


def test_admit_waiting_sequences_returns_empty_when_nothing_is_waiting(
    scheduler: Scheduler,
) -> None:
    """Return no admissions when the waiting queue is empty."""
    admitted = scheduler._admit_waiting_sequences()

    assert admitted == ()
    assert scheduler.running_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_admit_sequence_rejects_non_head_sequence_without_mutation(
    scheduler: Scheduler,
) -> None:
    """Reject out-of-order admission before allocating KV-cache blocks."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(2,),
            max_new_tokens=8,
        ),
    )

    second_sequence = scheduler.waiting_sequences[1]

    with pytest.raises(
        RuntimeError,
        match="Admission violated waiting-queue ordering",
    ):
        scheduler._admit_sequence(second_sequence)

    assert scheduler.block_pool.num_allocated_blocks == 0

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == [
        "request-1",
        "request-2",
    ]


# Step scheduling


def test_schedule_returns_empty_output_when_no_work_exists(
    scheduler: Scheduler,
) -> None:
    """Return an empty scheduling decision when no work exists."""
    output = scheduler.schedule()

    assert output.decode_sequences == ()
    assert output.prefill_sequences == ()
    assert output.num_batched_tokens == 0


def test_schedule_admits_waiting_requests_as_prefill() -> None:
    """Admit waiting requests as prefill work within the token budget."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_sequences=4,
        max_batched_tokens=8,
    )
    scheduler = Scheduler(configuration=configuration)

    request_id1, request_id2 = "request-1", "request-2"
    max_new_tokens = 1

    scheduler.submit(
        Request(
            request_id=request_id1,
            prompt_token_ids=(1, 2, 3),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id=request_id2,
            prompt_token_ids=(4, 5, 6, 7),
            max_new_tokens=max_new_tokens,
        ),
    )

    output = scheduler.schedule()

    assert output.decode_sequences == ()

    assert [sequence.request.request_id for sequence in output.prefill_sequences] == [
        request_id1,
        request_id2,
    ]

    assert output.num_batched_tokens == 7
    assert scheduler.waiting_sequences == ()

    assert [
        sequence.request.request_id for sequence in scheduler.running_sequences
    ] == [request_id1, request_id2]


def test_schedule_accounts_for_current_length_during_prefill() -> None:
    """Charge the full current sequence history against the prefill token budget."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=4,
        max_sequences=2,
        max_batched_tokens=8,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1, 2, 3, 4),
            max_new_tokens=5,
        ),
    )

    sequence = scheduler.waiting_sequences[0]
    sequence.generated_token_ids.extend([5, 6, 7])

    output = scheduler.schedule()

    assert sequence.current_length == 7
    assert output.decode_sequences == ()
    assert output.prefill_sequences == (sequence,)
    assert output.num_batched_tokens == 7


def test_schedule_selects_existing_running_sequences_as_decode(
    scheduler: Scheduler,
) -> None:
    """Select already admitted sequences as decode work."""
    max_new_tokens = 8

    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(2,),
            max_new_tokens=max_new_tokens,
        ),
    )

    admitted = scheduler._admit_waiting_sequences()

    output = scheduler.schedule()

    assert output.decode_sequences == admitted
    assert output.prefill_sequences == ()
    assert output.num_batched_tokens == 2


def test_schedule_prioritizes_decode_before_prefill() -> None:
    """Consume decode budget before admitting waiting prefill work."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_sequences=4,
        max_batched_tokens=6,
    )
    scheduler = Scheduler(configuration=configuration)

    request_decode_id1, request_decode_id2 = "decode-1", "decode-2"
    request_prefill_id1, request_prefill_id2 = "prefill-1", "prefill-2"
    max_new_tokens = 3

    scheduler.submit(
        Request(
            request_id=request_decode_id1,
            prompt_token_ids=(1,),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id=request_decode_id2,
            prompt_token_ids=(2,),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler._admit_waiting_sequences()

    scheduler.submit(
        Request(
            request_id=request_prefill_id1,
            prompt_token_ids=(3, 4, 5, 6),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id=request_prefill_id2,
            prompt_token_ids=(7,),
            max_new_tokens=max_new_tokens,
        ),
    )

    output = scheduler.schedule()

    assert [sequence.request.request_id for sequence in output.decode_sequences] == [
        "decode-1",
        "decode-2",
    ]

    assert [sequence.request.request_id for sequence in output.prefill_sequences] == [
        "prefill-1",
    ]

    assert output.num_batched_tokens == 6

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["prefill-2"]


def test_schedule_preserves_fcfs_when_head_prefill_exceeds_remaining_budget() -> None:
    """Do not skip the queue head when its prompt exceeds remaining budget."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_sequences=4,
        max_batched_tokens=8,
    )
    scheduler = Scheduler(configuration=configuration)

    running_req = "running-request"
    large_req = "large-request"
    small_req = "small-request"
    max_new_tokens = 1

    scheduler.submit(
        Request(
            request_id=running_req,
            prompt_token_ids=(1,),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler._admit_waiting_sequences()

    scheduler.submit(
        Request(
            request_id=large_req,
            prompt_token_ids=tuple(range(8)),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id=small_req,
            prompt_token_ids=(1,),
            max_new_tokens=max_new_tokens,
        ),
    )

    output = scheduler.schedule()

    assert [sequence.request.request_id for sequence in output.decode_sequences] == [
        running_req
    ]

    assert output.prefill_sequences == ()
    assert output.num_batched_tokens == 1

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == [large_req, small_req]


def test_schedule_stops_prefill_admission_at_max_sequences() -> None:
    """Stop prefill admission when running sequence capacity is reached."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=8,
        max_sequences=2,
        max_batched_tokens=8,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="running-request",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )
    scheduler._admit_waiting_sequences()

    scheduler.submit(
        Request(
            request_id="waiting-1",
            prompt_token_ids=(2,),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="waiting-2",
            prompt_token_ids=(3,),
            max_new_tokens=8,
        ),
    )

    output = scheduler.schedule()

    assert [sequence.request.request_id for sequence in output.decode_sequences] == [
        "running-request",
    ]

    assert [sequence.request.request_id for sequence in output.prefill_sequences] == [
        "waiting-1",
    ]

    assert output.num_batched_tokens == 2

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["waiting-2"]

    assert len(scheduler.running_sequences) == 2


def test_waiting_request_is_admitted_after_running_request_releases_blocks() -> None:
    """Admit a blocked request after KV capacity is released."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_sequences=2,
        max_batched_tokens=32,
    )
    scheduler = Scheduler(configuration=configuration)

    max_new_tokens = 1

    scheduler.submit(
        Request(
            request_id="request-a",
            prompt_token_ids=tuple(range(32)),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-b",
            prompt_token_ids=tuple(range(32)),
            max_new_tokens=max_new_tokens,
        ),
    )

    first_output = scheduler.schedule()

    assert [
        sequence.request.request_id for sequence in first_output.prefill_sequences
    ] == ["request-a"]

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == ["request-b"]

    first_sequence = scheduler.running_sequences[0]

    scheduler.finish_sequence(
        sequence=first_sequence,
        reason=FinishReason.EOS,
    )

    assert scheduler.block_pool.num_free_blocks == 2

    second_output = scheduler.schedule()

    assert [
        sequence.request.request_id for sequence in second_output.prefill_sequences
    ] == ["request-b"]

    assert scheduler.waiting_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 2


# Decode KV growth


def test_reserve_generated_token_reuses_existing_block(scheduler: Scheduler) -> None:
    """Reuse the current KV block when the next token fits inside it."""
    request_id = "request-1"

    scheduler.submit(
        Request(
            request_id=request_id,
            prompt_token_ids=tuple(range(15)),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    assert scheduler.block_table.blocks_for_request(request_id) == (0,)
    assert scheduler.block_pool.num_allocated_blocks == 1

    scheduler.reserve_generated_token(sequence=sequence)

    assert scheduler.block_table.blocks_for_request(request_id) == (0,)
    assert scheduler.block_pool.num_allocated_blocks == 1
    assert scheduler.block_pool.num_free_blocks == 3


def test_reserve_generated_token_allocates_block_at_boundary(
    scheduler: Scheduler,
) -> None:
    """Allocate one new KV block when generation crosses a block boundary."""
    request_id = "request-1"

    scheduler.submit(
        Request(
            request_id=request_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    assert scheduler.block_table.blocks_for_request(request_id) == (0,)
    assert scheduler.block_pool.num_allocated_blocks == 1
    assert scheduler.block_pool.num_free_blocks == 3

    scheduler.reserve_generated_token(sequence)

    assert scheduler.block_table.blocks_for_request(request_id) == (0, 1)
    assert scheduler.block_pool.num_allocated_blocks == 2
    assert scheduler.block_pool.num_free_blocks == 2


def test_reserve_generated_token_is_idempotent_for_existing_mapping(
    scheduler: Scheduler,
) -> None:
    """Avoid duplicate allocation when the same logical block is reserved again."""
    request_id = "request-1"

    scheduler.submit(
        Request(
            request_id=request_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    scheduler.reserve_generated_token(sequence)
    scheduler.reserve_generated_token(sequence)

    assert scheduler.block_table.blocks_for_request(request_id) == (0, 1)
    assert scheduler.block_pool.num_allocated_blocks == 2
    assert scheduler.block_pool.num_free_blocks == 2


def test_reserve_generated_token_rejects_non_running_sequence(
    scheduler: Scheduler,
) -> None:
    """Reject KV reservation for a sequence that has not been admitted."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler.waiting_sequences[0]

    with pytest.raises(RuntimeError, match="sequence that is not running"):
        scheduler.reserve_generated_token(sequence)

    assert scheduler.block_pool.num_allocated_blocks == 0


def test_reserve_generated_token_rejects_finished_sequence(
    scheduler: Scheduler,
) -> None:
    """Reject KV reservation after a sequence has finished."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    scheduler.finish_sequence(
        sequence=sequence,
        reason=FinishReason.EOS,
    )

    with pytest.raises(RuntimeError, match="storage for a finished sequence"):
        scheduler.reserve_generated_token(sequence)

    assert scheduler.block_pool.num_allocated_blocks == 0


def test_reserve_generated_token_preempts_victim_and_retries() -> None:
    """Preempt another running sequence and retry KV reservation."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_sequences=2,
        max_batched_tokens=32,
    )
    scheduler = Scheduler(configuration=configuration)

    requester_id = "requester"
    victim_id = "victim"

    scheduler.submit(
        Request(
            request_id=requester_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=2,
        ),
    )
    scheduler.submit(
        Request(
            request_id=victim_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=1,
        ),
    )

    requester, victim = scheduler._admit_waiting_sequences()

    assert scheduler.block_table.blocks_for_request(requester_id) == (0,)
    assert scheduler.block_table.blocks_for_request(victim_id) == (1,)
    assert scheduler.block_pool.num_free_blocks == 0

    scheduler.reserve_generated_token(requester)

    assert scheduler.num_preemptions == 1
    assert requester.status == SequenceStatus.RUNNING
    assert victim.status == SequenceStatus.PREEMPTED

    assert scheduler.running_sequences == (requester,)
    assert scheduler.waiting_sequences == (victim,)

    assert scheduler.block_table.blocks_for_request(requester_id) == (0, 1)

    with pytest.raises(ValueError, match="Unknown request_id"):
        scheduler.block_table.blocks_for_request(victim_id)

    assert scheduler.block_pool.num_allocated_blocks == 2
    assert scheduler.block_pool.num_free_blocks == 0

    assert_scheduler_invariant(scheduler)


def test_reserve_generated_token_does_not_preempt_when_block_is_reused() -> None:
    """Avoid preemption when reservation needs no additional KV block."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_sequences=2,
        max_batched_tokens=32,
    )
    scheduler = Scheduler(configuration=configuration)

    requester_id, other_request_id = "requester", "other-request"

    scheduler.submit(
        Request(
            request_id=requester_id,
            prompt_token_ids=tuple(range(15)),
            max_new_tokens=2,
        ),
    )
    scheduler.submit(
        Request(
            request_id=other_request_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=1,
        ),
    )

    requester, other_sequence = scheduler._admit_waiting_sequences()

    assert scheduler.block_pool.num_free_blocks == 0

    scheduler.reserve_generated_token(requester)

    assert scheduler.running_sequences == (requester, other_sequence)
    assert scheduler.waiting_sequences == ()

    assert requester.status == SequenceStatus.RUNNING
    assert other_sequence.status == SequenceStatus.RUNNING

    assert scheduler.block_table.blocks_for_request(requester_id) == (0,)
    assert scheduler.block_table.blocks_for_request(other_request_id) == (1,)

    assert scheduler.block_pool.num_free_blocks == 0


# Preemption


def test_preemption_count_tracks_successful_preemptions(
    scheduler: Scheduler,
) -> None:
    """Count each successful sequence preemption."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(2,),
            max_new_tokens=8,
        ),
    )

    first_sequence, second_sequence = scheduler._admit_waiting_sequences()

    assert scheduler.num_preemptions == 0

    scheduler._preempt_sequence(second_sequence)

    assert scheduler.num_preemptions == 1

    scheduler._preempt_sequence(first_sequence)

    assert scheduler.num_preemptions == 2


def test_preempt_sequence_releases_kv_and_moves_to_waiting_front(
    scheduler: Scheduler,
) -> None:
    """Release victim KV blocks and move the sequence to the waiting front."""
    victim_request_id = "victim"
    already_waiting_request_id = "already-waiting"
    max_new_tokens = 8

    scheduler.submit(
        Request(
            request_id=victim_request_id,
            prompt_token_ids=tuple(range(17)),
            max_new_tokens=max_new_tokens,
        ),
    )

    victim = scheduler._admit_waiting_sequences()[0]

    scheduler.submit(
        Request(
            request_id=already_waiting_request_id,
            prompt_token_ids=(1,),
            max_new_tokens=max_new_tokens,
        ),
    )

    assert scheduler.block_table.blocks_for_request(victim_request_id) == (0, 1)
    assert scheduler.block_pool.num_allocated_blocks == 2

    scheduler._preempt_sequence(victim)

    assert_scheduler_invariant(scheduler)

    assert victim.status == SequenceStatus.PREEMPTED
    assert scheduler.running_sequences == ()

    assert [
        sequence.request.request_id for sequence in scheduler.waiting_sequences
    ] == [
        victim_request_id,
        already_waiting_request_id,
    ]

    assert scheduler.block_pool.num_allocated_blocks == 0
    assert scheduler.block_pool.num_free_blocks == 4

    with pytest.raises(ValueError, match="Unknown request_id"):
        scheduler.block_table.blocks_for_request(victim_request_id)


def test_preempt_sequence_preserves_generated_history(
    scheduler: Scheduler,
) -> None:
    """Preserve generated tokens and logical length across preemption."""
    request_id1 = "request-1"

    scheduler.submit(
        Request(
            request_id=request_id1,
            prompt_token_ids=(1, 2, 3),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    sequence.generated_token_ids.extend([4, 5, 6])

    generated_before = tuple(sequence.generated_token_ids)
    current_length_before = sequence.current_length

    scheduler._preempt_sequence(sequence)

    assert sequence.status == SequenceStatus.PREEMPTED
    assert tuple(sequence.generated_token_ids) == generated_before
    assert sequence.current_length == current_length_before
    assert sequence.finish_reason is None

    assert scheduler.waiting_sequences == (sequence,)
    assert scheduler.running_sequences == ()

    with pytest.raises(ValueError, match="Unknown request_id"):
        scheduler.block_table.blocks_for_request(request_id1)


def test_preempted_sequence_is_readmitted_after_kv_becomes_available() -> None:
    """Readmit a preempted sequence after the requester releases KV capacity."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=2,
        max_sequences=2,
        max_batched_tokens=32,
    )
    scheduler = Scheduler(configuration=configuration)

    victim_request_id = "victim"

    scheduler.submit(
        Request(
            request_id="requester",
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=2,
        ),
    )
    scheduler.submit(
        Request(
            request_id=victim_request_id,
            prompt_token_ids=tuple(range(16)),
            max_new_tokens=1,
        ),
    )

    requester, victim = scheduler._admit_waiting_sequences()

    scheduler.reserve_generated_token(requester)

    assert victim.status == SequenceStatus.PREEMPTED
    assert len(scheduler.waiting_sequences) == 1
    assert scheduler.waiting_sequences[0] is victim

    scheduler.finish_sequence(
        sequence=requester,
        reason=FinishReason.LENGTH,
    )

    output = scheduler.schedule()

    assert output.prefill_sequences == (victim,)
    assert len(scheduler.waiting_sequences) == 0
    assert len(scheduler.running_sequences) == 1
    assert scheduler.running_sequences[0] is victim
    assert scheduler.running_sequences[0].status == SequenceStatus.RUNNING

    assert scheduler.block_table.blocks_for_request(victim_request_id) == (0,)
    assert_scheduler_invariant(scheduler)


def test_preemption_victim_is_last_running_sequence(
    scheduler: Scheduler,
) -> None:
    """Select the most recently admitted running sequence as the victim."""
    max_new_tokens = 8

    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=max_new_tokens,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(2,),
            max_new_tokens=max_new_tokens,
        )
    )

    first_sequence, second_sequence = scheduler._admit_waiting_sequences()

    victim = scheduler._select_preemption_victim(
        requester=first_sequence,
    )

    assert victim is second_sequence


def test_preemption_victim_skips_requester_at_running_tail(
    scheduler: Scheduler,
) -> None:
    """Select the previous running sequence when the requester is last."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(2,),
            max_new_tokens=8,
        ),
    )

    first_sequence, second_sequence = scheduler._admit_waiting_sequences()

    victim = scheduler._select_preemption_victim(
        requester=second_sequence,
    )

    assert victim is first_sequence


def test_preemption_victim_excludes_requester(scheduler: Scheduler) -> None:
    """Never select the requesting sequence as its own preemption victim."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    requester = scheduler._admit_waiting_sequences()[0]

    victim = scheduler._select_preemption_victim(
        requester=requester,
    )

    assert victim is None


# Completion and release


def test_finish_sequence_releases_blocks_and_moves_sequence_to_finished(
    scheduler: Scheduler,
) -> None:
    """Release KV blocks and move a running sequence to finished."""
    request = Request(
        request_id="request-1",
        prompt_token_ids=tuple(range(17)),
        max_new_tokens=8,
    )

    scheduler.submit(request)
    admitted = scheduler._admit_waiting_sequences()

    sequence = admitted[0]

    assert scheduler.block_pool.num_allocated_blocks == 2

    scheduler.finish_sequence(
        sequence=sequence,
        reason=FinishReason.EOS,
    )

    assert scheduler.running_sequences == ()
    assert scheduler.finished_sequences == (sequence,)

    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.finish_reason == FinishReason.EOS

    assert scheduler.block_pool.num_allocated_blocks == 0
    assert scheduler.block_pool.num_free_blocks == 4

    with pytest.raises(ValueError, match="Unknown request_id"):
        scheduler.block_table.blocks_for_request("request-1")


@pytest.mark.parametrize(
    "finish_reason",
    [
        FinishReason.EOS,
        FinishReason.LENGTH,
    ],
)
def test_finish_sequence_preserves_finish_reason(
    scheduler: Scheduler,
    finish_reason: FinishReason,
) -> None:
    """Preserve the reason supplied when a running sequence finishes."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    scheduler.finish_sequence(
        sequence=sequence,
        reason=finish_reason,
    )

    assert sequence.finish_reason == finish_reason


def test_finish_sequence_preserves_other_running_request_blocks() -> None:
    """Release one request without affecting another running request."""
    configuration = EngineConfiguration(
        block_size=16,
        num_blocks=4,
        max_sequences=2,
        max_batched_tokens=64,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=tuple(range(17)),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    admitted = scheduler._admit_waiting_sequences()

    first_sequence = admitted[0]
    second_sequence = admitted[1]

    assert scheduler.block_table.blocks_for_request("request-1") == (0, 1)
    assert scheduler.block_table.blocks_for_request("request-2") == (2,)

    scheduler.finish_sequence(
        sequence=first_sequence,
        reason=FinishReason.EOS,
    )

    assert scheduler.running_sequences == (second_sequence,)
    assert scheduler.finished_sequences == (first_sequence,)

    assert scheduler.block_table.blocks_for_request("request-2") == (2,)

    assert scheduler.block_pool.num_allocated_blocks == 1
    assert scheduler.block_pool.num_free_blocks == 3


def test_finish_sequence_rejects_non_running_sequence(
    scheduler: Scheduler,
) -> None:
    """Reject finishing a sequence that has not been admitted."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler.waiting_sequences[0]

    with pytest.raises(
        RuntimeError,
        match="Cannot finish a sequence that is not running",
    ):
        scheduler.finish_sequence(
            sequence=sequence,
            reason=FinishReason.EOS,
        )

    assert scheduler.waiting_sequences == (sequence,)
    assert scheduler.finished_sequences == ()
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_finish_sequence_rejects_already_finished_sequence(
    scheduler: Scheduler,
) -> None:
    """Reject finishing the same sequence more than once."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=(1,),
            max_new_tokens=8,
        ),
    )

    sequence = scheduler._admit_waiting_sequences()[0]

    scheduler.finish_sequence(
        sequence=sequence,
        reason=FinishReason.EOS,
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot finish a sequence that is already finished",
    ):
        scheduler.finish_sequence(
            sequence=sequence,
            reason=FinishReason.LENGTH,
        )

    assert scheduler.finished_sequences == (sequence,)
    assert scheduler.block_pool.num_allocated_blocks == 0


def test_finishing_all_running_sequences_restores_full_pool(
    scheduler: Scheduler,
) -> None:
    """Return every KV-cache block after all running sequences finish."""
    scheduler.submit(
        Request(
            request_id="request-1",
            prompt_token_ids=tuple(range(17)),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-2",
            prompt_token_ids=tuple(range(17)),
            max_new_tokens=8,
        ),
    )

    admitted = scheduler._admit_waiting_sequences()

    assert scheduler.block_pool.num_allocated_blocks == 4
    assert scheduler.block_pool.num_free_blocks == 0

    for sequence in admitted:
        scheduler.finish_sequence(
            sequence=sequence,
            reason=FinishReason.LENGTH,
        )

    assert scheduler.running_sequences == ()
    assert len(scheduler.finished_sequences) == 2

    assert scheduler.block_pool.num_allocated_blocks == 0
    assert scheduler.block_pool.num_free_blocks == 4

    assert scheduler.has_unfinished_requests is False


# Scheduler invariants


def test_scheduler_invariant_through_mixed_request_lifecycle() -> None:
    """Preserve queue and KV ownership invariants through lifecycle changes."""
    configuration = EngineConfiguration(
        block_size=4,
        num_blocks=6,
        max_sequences=3,
        max_batched_tokens=12,
    )
    scheduler = Scheduler(configuration=configuration)

    scheduler.submit(
        Request(
            request_id="request-a",
            prompt_token_ids=(1, 2, 3, 4),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-b",
            prompt_token_ids=(5, 6),
            max_new_tokens=8,
        ),
    )
    scheduler.submit(
        Request(
            request_id="request-c",
            prompt_token_ids=(7,),
            max_new_tokens=8,
        ),
    )

    assert_scheduler_invariant(scheduler)

    scheduler.schedule()
    assert_scheduler_invariant(scheduler)

    sequence = scheduler.running_sequences[0]

    scheduler.reserve_generated_token(sequence)
    assert_scheduler_invariant(scheduler)

    scheduler.finish_sequence(
        sequence=sequence,
        reason=FinishReason.EOS,
    )
    assert_scheduler_invariant(scheduler)
