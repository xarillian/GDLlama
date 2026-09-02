# ADR-002: Keep inference asynchronous and fence every request lifecycle

Status: Accepted
Date: 2026-07-11

## Context

Blocking generation methods and compatibility aliases duplicated the provider's asynchronous behavior while making cancellation, replacement, and host-thread delivery harder to reason about. Replacing a live engine can also orphan requests or temporarily hold two models in scarce GPU memory. Conversation history belongs to the same application lifecycle, not to one host binding.

## Decision

Expose generation and embeddings only through asynchronous requests and polled runtime events. `ChorusRuntime` owns request identity, conversation history, and terminal delivery. Every accepted request produces exactly one terminal event.

Stopping or replacing an engine cancels each live request exactly once and fences provider callbacks before teardown completes. Engine replacement is destructive: stop and release the old engine before loading its successor. Hosts receive events on their own thread and do not call blocking compatibility methods.

## Consequences

Positive:

- Hosts share one request lifecycle and cancellation model.
- Provider callbacks cannot reach torn-down runtime or host state after shutdown.
- Replacement never requires two models to reside in GPU memory at once.
- Conversation behavior stays consistent across host adapters.

Negative:

- Simple callers must poll and correlate request identifiers even for one request.
- Replacing an engine interrupts all work on the old engine.
- No synchronous compatibility surface exists for consumers migrating from GDLlama.

## Compliance

- Each accepted request must terminate once with completion, error, or cancellation.
- Provider callbacks must enter a synchronized channel rather than host code.
- Provider shutdown must complete before the old engine or callback state is destroyed.
- Host adapters must translate the asynchronous runtime API without adding blocking inference paths.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial accepted record
