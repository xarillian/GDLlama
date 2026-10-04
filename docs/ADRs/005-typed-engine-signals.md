# ADR-005: Represent engine signals as typed variants

Status: Overwritten 2026-10-03 by [ADR-009](009-typed-terminal-signals.md)
Date: 2026-08-30

## Context

An engine signal previously paired an `EventType` discriminator with a structure containing fields that were valid only for some event kinds. This allowed contradictory states, required consumers to know which fields were meaningful, and duplicated the discriminator in both the type tag and payload conventions.

## Decision

`ChorusSignal` carries a `std::variant` of token, embedding, stop, and error events. Each event type contains only its valid payload:

- Token events contain a channel and text.
- Embedding events contain values.
- Error events contain an error code and message.
- Stop events contain no payload.

Delete `EventType` from the provider contract rather than retaining a second discriminator. The runtime translates typed engine signals into the host-facing `RuntimeEvent` vocabulary.

## Consequences

Positive:

- Invalid combinations of event kind and payload are not representable.
- Visitors and exhaustive handling expose missing cases when the contract grows.
- Providers construct only the data required by the emitted event.

Negative:

- Adding a signal kind changes the public variant and every exhaustive visitor.
- Provider and runtime code must use variant construction and visitation.
- Engine signals and host-facing runtime events remain separate vocabularies with an explicit translation step.

## Compliance

- Provider implementations must emit one concrete `ChorusSignal` alternative.
- No parallel event discriminator or conditionally valid payload structure may be added.
- Runtime translation must handle every `ChorusSignal` alternative explicitly.
- Provider contract tests must cover the payload shape of each alternative.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial accepted record
