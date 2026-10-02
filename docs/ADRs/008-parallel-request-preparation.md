# ADR-008: Prepare requests in parallel and publish them in admission order

Status: Proposed
Date: 2026-09-30

## Context

Each loaded runtime prepared requests on one worker. Preparation renders a conversation through the model's chat template and tokenizes it, which is CPU work that grows with history length. When many agents answered one event, every request waited for every render before it: with Gemma 4 E4B, 40 requests with ~3,900-token histories spent about 9 of 10 seconds queued behind renders of roughly 225 ms each, while the GPU sat mostly idle.

## Decision

Each loaded runtime runs a small pool of preparation workers, `min(4, hardware threads / 4)` and at least one, fed by the existing bounded FIFO queue. Workers take a ticket when they take a job and may finish in any order; outcomes, including errors and preview or count results, publish strictly in ticket order. `Chorus::RequestPreparation` implementations must be safe for concurrent calls.

## Consequences

Positive:

- A burst of long conversations prepares in parallel, removing most of the queueing in front of the engine.
- Equal-priority requests still reach the engine in submission order, and observable event order is unchanged.

Negative:

- Providers must make preparation thread-safe; the llama provider keeps one parsed template set per concurrent render because llama.cpp does not guarantee concurrent use of one parsed template.
- A slow request still holds back publication of faster ones admitted after it.
- The order in which a provider receives preparation calls is no longer defined.

## Compliance

- Preparation outcomes publish only in admission order.
- A worker that fails must still complete its ticket so later outcomes are not stranded.
- Replacement and shutdown join every preparation worker before releasing provider resources.
- Tests that depend on work remaining queued fix the worker count through `ChorusRuntime::test_use_preparation_workers`.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial accepted record
