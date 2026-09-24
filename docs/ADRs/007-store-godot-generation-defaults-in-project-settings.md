# ADR-007: Store Godot generation defaults in ProjectSettings

Status: TODO
Date: 2026-09-09

## Context

- Godot projects need shared generation defaults in a native project-wide location.
- Stored choices must distinguish absence from valid false, zero, empty string, and empty collection values.

## Decision

- Store one project-wide generation-default set in `ProjectSettings`.
- Resolve each option in request, project, provider order as defined by [ADR-006](006-express-public-generation-configuration-as-caller-intent.md).
- Keep `ProjectSettings` access inside the Godot adapter.
- Expose no project-default facility through the C ABI. An application built over it. e.g. unity integration, owns that decision.

## Consequences

Positive:

- `GodotChorus` nodes share one default set without per-node default resources.

Negative:

- Different ambient default sets require request choices or a later named-profile decision.

## Compliance

- Explicit false, zero, and empty values survive project reload.
- Runtime and provider code perform no `ProjectSettings` I/O.

## Notes

- Version: 0.1
- Changelog:
  - 0.1: Initial proposed record
