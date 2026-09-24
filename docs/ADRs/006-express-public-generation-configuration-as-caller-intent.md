# ADR-006: Express public generation configuration as caller intent

Status: TODO
Date: 2026-09-08

## Context

A generation option may come from the provider, host defaults, or one request. The current generation API exposes that composition through `PatchAction`, `ConfigPatch<T>`, `GenerationConfigPatch`, and the Godot `ChorusOverrideState` enum.

A Godot caller can assign `request.max_tokens` without affecting generation until they separately mark the value as `SET`. The caller must choose between `INHERIT` and `CLEAR`, where `CLEAR` bypasses host defaults and `INHERIT` removes only the request choice. These names describe how Chorus merges layers rather than what the caller wants. They make ordinary property assignment inert and couple consumers to the number and provenance of configuration sources.

We considered retaining the public three-action patch model, but it preserves that coupling and its surprising assignment behavior. We considered making request configuration replace the complete host configuration, but callers would have to repeat unrelated host choices. We instead use presence-aware choices at each caller scope, compose those scopes in the runtime, and leave provider defaults with the provider.

## Decision

Public generation configuration expresses choices at the caller's current scope, not the mechanics used to compose those choices with other scopes.

An option at one public scope has two states:

- A value is present, so Chorus uses that value at that scope.
- A value is absent, so Chorus continues normal default resolution.

For each option, a request value takes precedence over a host-default value. When neither caller scope supplies a value, the provider applies its own default. The runtime composes only the caller-owned scopes; it neither discovers nor copies provider defaults. A provider receives effective set of caller choices and does not receive separate request and host layers.

Assignment takes effect without a separate activation action. Clearing removes only the choice from the object being changed and resumes normal default resolution. A request cannot generically bypass a host-default value to reveal the provider value beneath it.

An explicit false, zero, empty string, or empty collection remains a value when valid for that option. Public types preserve the distinction between such a value and absence.

Provider namespaces combine structurally, but each option key within a namespace is one value. A present scalar, string, list, or map replaces the lower-scope value in full. An empty list or map is therefore an explicit empty value, not a request to retain lower entries. Public APIs do not expose recursive merge actions or lower-layer erasure lists.

The runtime accepts host defaults as injected data and performs no persistence or host-configuration I/O. Presentation layers that expose host defaults choose an idiomatic source and translate its values into the host-neutral runtime contract. Godot project defaults are governed by [ADR-007](007-store-godot-generation-defaults-in-project-settings.md).

If consumers later need behavior that cannot be expressed as set or use-default, that behavior must earn a domain name and contract of its own. Chorus does not expose a generic escape hatch for selecting a configuration source.

## Consequences

Positive:

- Property assignment and clearing match ordinary caller expectations.
- Consumers remain independent of internal composition mechanics.
- Host adapters can present idiomatic APIs while preserving one cross-host meaning.
- Provider defaults remain owned by each provider.
- Validation applies only to choices that are present and effective at that scope.

Negative:

- False, zero, empty strings, and empty collections require presence-aware storage so they remain distinct from absence.
- A request cannot select the provider default beneath a host-default value without supplying a concrete value.
- A collection-valued provider option replaces the complete lower-scope collection rather than deep-merging selected entries.
- Behavior for an absent option may change when its provider changes that default.

## Compliance

- Every writable public generation property takes effect when assigned without a second activation call.
- A public clear operation removes only the choice owned by the object receiving that operation.
- Request choices take precedence over host defaults, and absence at both scopes delegates to the provider.
- A request cannot bypass a host default without supplying a concrete domain value.
- Public request APIs do not expose merge-action enums, activation state properties, lower-layer erasure lists, or methods named for inheritance.
- Public option representations distinguish absence from valid false, zero, empty strings, and empty collections where the domain permits them.
- A present provider-option list or map replaces the complete value at that option key, including when empty.
- The runtime composes caller-owned scopes and performs no settings-file or database I/O.
- Providers own their defaults and receive no separate request and host configuration layers.
- Host adapters translate host-native set and clear operations into runtime intent without exposing runtime composition mechanics.
- Tests cover precedence, assignment, local clearing, provider fallback, and explicit false, zero, and empty values.
- Reviews reject new public generation controls whose names describe layer provenance or merge mechanics unless a separate accepted decision establishes them as domain concepts.

## Notes

- Version: 0.3
- Changelog:
  - 0.3: Narrow the decision to generation configuration, clarify precedence and provider ownership, define collection replacement, and move Godot persistence to ADR-007
  - 0.2: Assign default persistence to hosts, select Godot `ProjectSettings`, and move named profiles to the backlog
  - 0.1: Initial proposed record
