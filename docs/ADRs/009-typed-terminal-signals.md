# ADR-009: End every request with a typed terminal signal that carries its result

Status: Accepted
Date: 2026-10-03

## Context

[ADR-005](005-typed-engine-signals.md) made engine signals a typed variant in which each alternative carries only its valid payload:

- `Chorus::ChorusSignal::Token` carries a channel and text.
- `Chorus::ChorusSignal::Embedding` carries a vector.
- `Chorus::ChorusSignal::Error` carries an error code and message.
- `Chorus::ChorusSignal::Stop` carries nothing.

Because the success terminal carries nothing, results must travel in separate signals ahead of it. An embedding request emits `Chorus::ChorusSignal::Embedding` and then `Chorus::ChorusSignal::Stop`, and the runtime must reject a `Chorus::ChorusSignal::Stop` that arrives without a vector.

Generations now need terminal data too, starting with token usage, and the contract has no place for it. Following the embedding pattern would give each new fact its own signal, which must arrive exactly once, after the last token and before `Chorus::ChorusSignal::Stop`. Each such rule needs its own runtime check and contract test.

The host vocabulary already pairs results with terminals. `Chorus::RuntimeEvent::Kind::Complete`, `Chorus::RuntimeEvent::Kind::Embedding` and `Chorus::RuntimeEvent::Kind::Error` each end a request and carry its outcome.

## Decision

Every accepted request ends in exactly one terminal alternative, and that alternative carries the request's result:

- `Chorus::ChorusSignal::Completion` ends a generation and carries its terminal data, starting with token usage.
- `Chorus::ChorusSignal::Embedding` ends an embedding request and carries its vector.
- `Chorus::ChorusSignal::Error` ends any request in failure.

`Chorus::ChorusSignal::Token` is the only alternative that does not end a request. Delete `Chorus::ChorusSignal::Stop`. Data that describes a finished request belongs on its terminal, never in a separate signal.

This record replaces ADR-005 and keeps its rules: each alternative carries only its valid payload, and no discriminator or conditionally valid field may be added.

## Consequences

Positive:

- A provider ends a request by emitting one signal. No ordering rule beyond "tokens, then one terminal" exists to break.
- A missing or misplaced result cannot be expressed, so the runtime checks and contract cases that guard against them disappear.
- Runtime translation maps each terminal to one host event.
- New terminal data, such as the reason generation stopped, extends a terminal's payload instead of adding a signal.

Negative:

- Every new request kind needs its own terminal alternative.
- Changing a terminal's payload changes every provider that emits it.
- A failed or cancelled request reports nothing but its error. Partial results such as usage would need a new failure alternative, since a field on `Chorus::ChorusSignal::Error` would be valid only for some failures.
- The cutover reaches every provider, the runtime, test doubles and every test that waits for `Chorus::ChorusSignal::Stop`.

## Compliance

- Every accepted request emits exactly one terminal: `Chorus::ChorusSignal::Completion`, `Chorus::ChorusSignal::Embedding` or `Chorus::ChorusSignal::Error`.
- Only `Chorus::ChorusSignal::Token` precedes a terminal, and nothing follows one.
- Every signal must suit its request: `Chorus::ChorusSignal::Token` and `Chorus::ChorusSignal::Completion` belong to generations, `Chorus::ChorusSignal::Embedding` to embedding requests. The runtime ends a mismatched request with `Chorus::ChorusError::Unknown`.
- Runtime translation handles every alternative explicitly.
- Provider contract tests cover each alternative's payload.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial accepted record
