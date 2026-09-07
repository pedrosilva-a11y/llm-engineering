# LLM Inference Engine

This package implements a small LLM inference engine focused on understanding the
control flow and resource-management decisions behind production serving systems.

The implementation intentionally builds scheduling, paged KV-cache management,
preemption, batching, and generation control explicitly rather than delegating
those responsibilities to an existing serving framework.

## Architecture

The engine is divided into three main responsibilities:

* `Engine` coordinates model execution and generated-token handling.
* `Scheduler` owns request admission, execution ordering, and KV-cache lifecycle.
* `ModelRunner` performs batched model execution.

Supporting components include:

* `SequenceState` for mutable request-generation state.
* `BlockPool` for physical KV-cache block ownership.
* `BlockTable` for logical-to-physical KV block mappings.
* `EngineConfiguration` for runtime capacity and scheduling limits.

The intended ownership boundary is:

```text
Engine
├── Scheduler
│   ├── waiting / running / finished lifecycle
│   ├── admission policy
│   ├── per-step token budget
│   └── KV-cache allocation and release
│
├── ModelRunner
│   └── batched model execution
│
└── SequenceState
    └── generated-token and lifecycle state
```

## Recompute preemption

When a running sequence requires additional KV capacity and the block pool is
exhausted, the scheduler may preempt another running sequence rather than
terminating generation.

Preemption uses **recomputation** rather than CPU swapping:

* generated tokens are preserved;
* KV-cache blocks are released;
* the sequence returns to the waiting queue;
* its KV state is reconstructed when it is later admitted again.

## Preemption trigger

Preemption is reactive.

The generation flow is:

```text
schedule
→ forward
→ select token
→ reserve KV for a continuing token
→ append token
```

If KV reservation fails after model execution, another eligible running sequence
is preempted and the reservation is retried.

This means a preemption event may waste computation already performed for the
current model step.

The choice is deliberate. It keeps model execution and KV growth consistent with
the existing engine boundary and makes the cost of reactive preemption directly
measurable.

A proactive alternative would reserve all required KV capacity during scheduling,
before model execution. That avoids wasted forward passes but requires the
scheduler to predict and reserve execution resources before the runner is called.

## Victim selection

The scheduler uses the **last-scheduled eligible running sequence** as the
preemption victim.

Conceptually:

```text
running = [A, B, C]

A requires additional KV
→ inspect candidates from the end
→ C is selected
```

The sequence requesting additional KV capacity is excluded from victim selection.

This policy:

* preserves more progress for older FCFS requests;
* is deterministic;
* provides a simple baseline for future policy comparisons.

Alternative policies could consider sequence length, blocks released, recompute
cost, or explicit priorities.

## Waiting-queue re-entry

A preempted sequence returns to the **front of the waiting queue**.

```text
waiting before preemption:
[D, E]

preempt B:
[B, D, E]
```

This preserves the sequence's earlier scheduling position rather than treating it
as a newly submitted request.

Being at the front does not bypass admission checks. The sequence is still
admitted only when execution and KV capacity are sufficient.

## Generated-token preservation

Preemption releases KV state but does not restart generation.

Before preemption:

```text
prompt:
[P1, P2, P3]

generated:
[G1, G2, G3]

KV:
allocated
```

After preemption:

```text
prompt:
[P1, P2, P3]

generated:
[G1, G2, G3]

KV:
released
```

When readmitted, the sequence reconstructs KV state from its complete token
history:

```text
[P1, P2, P3, G1, G2, G3]
```

Therefore readmission capacity must be calculated from `SequenceState.current_length`
rather than only from the original prompt length.

## Sequence lifecycle

Preemption introduces a distinct `PREEMPTED` lifecycle state.

The relevant transitions are:

```text
WAITING → RUNNING

RUNNING → FINISHED

RUNNING → PREEMPTED

PREEMPTED → RUNNING
```

A preempted sequence:

* retains its request;
* retains its generated tokens;
* has no finish reason;
* owns no KV-cache blocks;
* waits for readmission.

## KV capacity invariant

Any accepted request must be capable of fitting in the complete KV pool by itself
for every non-terminal point in its generation lifecycle.

Terminal tokens are appended to generated-token history but are never consumed by
another decode step, so they do not require additional KV storage.

The maximum KV-backed token count for one request is therefore:

```text
len(prompt_token_ids) + max_new_tokens - 1
```

Submission should reject requests whose maximum required KV capacity exceeds the
entire block pool.

This preserves the liveness property:

```text
if every other request releases its blocks,
an accepted request can always make progress by itself
```

It also guarantees that a preempted sequence can eventually be re-prefilled.

## Re-prefill sizing

KV block requirements belong to the scheduler.

`SequenceState` knows its token progress through `current_length`, while the
scheduler owns `block_size` and paged-KV policy.

Block sizing is therefore conceptually:

```text
blocks_required_for_sequence(sequence)
    = blocks_needed(
        sequence.current_length,
        configuration.block_size,
      )
```

The same calculation supports both fresh admission and readmission:

```text
fresh request:
current_length == prompt length

preempted request:
current_length == prompt length + generated tokens
```

This keeps `SequenceState` independent of KV paging details.

## KV release

Finishing and preempting both require the same low-level resource operation:

```text
release request KV blocks
→ remove sequence from running set
```

The lifecycle operation applied afterward differs:

```text
completion:
release
→ FINISHED
→ finished collection

preemption:
release
→ PREEMPTED
→ front of waiting queue
```

The shared release behavior should remain centralized so KV ownership stays
consistent across both paths.

## Self-preemption invariant

A sequence requesting additional KV capacity must never preempt itself.

Without this guard, a single sequence under KV pressure could repeatedly:

```text
require a block
→ preempt itself
→ release its blocks
→ get readmitted
→ require a block
→ preempt itself
→ ...
```

The requester is therefore excluded from victim selection.

## Scheduling invariants

The scheduler maintains the following ownership rules:

```text
WAITING
→ owns no KV blocks

RUNNING
→ owns KV blocks

PREEMPTED
→ owns no KV blocks

FINISHED
→ owns no KV blocks
```

Additionally:

* request IDs remain unique across the full scheduler lifecycle;
* existing decode work consumes scheduling budget before new prefill work;
* admission preserves FCFS head-of-line behavior;
* a model step performs zero or one batched runner call;
* terminal generated tokens do not allocate unnecessary future KV capacity.

## Preemption observability

Preemption counts should be observable.

At minimum, the scheduler should track aggregate preemptions:

```text
num_preemptions
```

This allows benchmark results to distinguish normal steady-state throughput from
memory-starved workloads experiencing recomputation.

Per-sequence counters can additionally support future starvation and fairness
analysis.

## Current limitations

The engine intentionally does not yet model several production-serving features:

* CPU KV-cache swapping;
* priority scheduling;
* sophisticated fairness policies;
* chunked prefill;
* prefix caching;
* real KV tensors;
* optimized recomputation;
* distributed execution.

These features can be layered onto the existing scheduler and lifecycle model
without changing the core ownership boundaries described above.

## Correctness goals

The implementation should preserve these properties:

* KV exhaustion during decode results in preemption rather than request failure.
* The requesting sequence can continue after another sequence releases capacity.
* Preempted sequences preserve generated tokens.
* Preempted sequences release all KV blocks.
* Readmission capacity is based on total current sequence length.
* A sequence cannot preempt itself.
* Oversubscribed workloads eventually complete when each individual request can
  fit in the pool.
* The KV block pool is fully restored after all requests finish.
* Preemption activity is observable.
* Workloads that do not require preemption retain equivalent generation behavior.
