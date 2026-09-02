# ADR-003: Point dependencies toward domain contracts

Status: Accepted
Date: 2026-07-18

## Context

Chorus must support more than one inference provider and more than one host without coupling either kind of integration to the other. Vendor libraries and host APIs change faster than the request, event, error, and lifecycle concepts shared across the product. Concrete providers also carry substantial link costs that should not reach inner-layer consumers or their tests.

## Decision

Organize Chorus as an inward-facing dependency graph around public domain contracts:

- Domain contracts define service interfaces and the value types they exchange.
- Application code orchestrates work through those interfaces and never names a concrete provider.
- Each infrastructure provider owns its vendor dependency and does not expose vendor types.
- The provider factory is the only catalog of concrete providers.
- Host adapters translate host idioms at the boundary and wire an engine into the application at their composition root.
- Generic utilities depend only on the standard library.

Capabilities and provider-owned option metadata describe provider differences. Consumers adapt to those declarations rather than branching on provider identity.

## Consequences

Positive:

- Providers and hosts can change independently.
- Core and runtime behavior can be built and tested without host or vendor dependencies.
- Adding a provider extends one catalog instead of spreading provider checks through consumers.

Negative:

- Data crossing a boundary sometimes needs an explicit public type or adapter conversion.
- Composition roots and factory targets carry extra wiring and build responsibilities.
- A feature that exposes a missing domain concept may require a deliberate contract change across several layers.

## Compliance

- Includes must follow the dependency graph documented in `docs/ARCHITECTURE.md`.
- Public headers must not include private, host, or vendor headers.
- Only the provider factory may name every concrete provider.
- Inner-layer test targets must build and link without host SDKs or vendor libraries.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial accepted record
