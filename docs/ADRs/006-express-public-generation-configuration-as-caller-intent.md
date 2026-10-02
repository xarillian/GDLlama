# ADR-006: Express public generation configuration as caller intent

Status: Proposed
Date: 2026-09-08

## Context

A generation option may come from the provider, host defaults, or one request. The current generation API exposes that composition through `Chorus::PatchAction`, `Chorus::ConfigPatch<T>`, `Chorus::GenerationConfigPatch`, and the Godot `ChorusOverrideState` enum.

A Godot caller can assign `request.max_tokens` without affecting generation until they separately mark the value as `ChorusOverrideState::SET`. The caller must choose between `ChorusOverrideState::INHERIT` and `ChorusOverrideState::CLEAR`, where clearing bypasses host defaults and inheriting removes only the request choice. These names describe how Chorus merges layers rather than what the caller wants. They make ordinary property assignment inert and couple consumers to the number and provenance of configuration sources.

We considered retaining the public three-action patch model, but it preserves that coupling and its surprising assignment behavior. We considered making request configuration replace the complete host configuration, but callers would have to repeat unrelated host choices. We instead use presence-aware choices at each caller scope, compose those scopes in the runtime, and leave provider defaults with the provider.

Recursive map merging makes partial changes convenient, but treats a supplied map as additions and replacements rather than a complete value. Without an erasure operation, an empty map cannot remove inherited entries. We compose at option boundaries instead, so a value's representation does not determine how it combines with defaults.

## Decision

Public generation configuration expresses choices at the caller's current scope, not the mechanics used to compose those choices with other scopes. Model paths, loading options, and their persistence are outside this decision's scope.

An option at one public scope has two presence states:

- A value is present, so Chorus uses that value at that scope.
- A value is absent, so Chorus continues normal default resolution.

For each option, a request value takes precedence over a host-default value. When neither caller scope supplies a value, the provider applies its own default. The runtime composes only the caller-owned scopes; it neither discovers nor copies provider defaults. A provider receives the effective set of caller choices, not separate request and host layers.

Assignment takes effect without a separate activation action. Clearing removes only the choice from the object being changed and resumes normal default resolution. A request cannot generically bypass a host-default value to reveal the provider value beneath it.

An explicit false, zero, empty string, or empty collection remains a value when valid for that option. The distinction between a value and absence survives runtime composition until the provider applies its defaults; absence must not become an empty or zero value at that boundary.

Output constraints follow the same presence rule:

- An absent request choice uses the default constraint.
- An explicitly unconstrained value selects generation without an output constraint.
- A grammar or schema selects that constraint as one complete value, including its format and source.

Unconstrained is a domain value, not absence or a merge action. It overrides a default constraint rather than asking for the provider's default. Clearing that choice restores normal default resolution. Providers must honor the explicit choice or reject it, never interpret it as absence.

Provider namespaces combine structurally, but each option key within a namespace is one value. Supplying an option leaves unrelated defaults intact. A present scalar, string, list, or map replaces the lower-scope value in full, including when the value is empty. An empty provider namespace supplies no option choices; an empty map at an option key is an explicit value. Public APIs do not expose recursive merge actions or lower-layer erasure lists.

The runtime accepts host defaults as injected data and performs no persistence or host-configuration I/O. Host adapters that expose defaults choose an idiomatic source and translate its values into the host-neutral runtime contract.

Other behavior beyond set or use-default must have a domain name and contract of its own. Chorus does not expose a generic escape hatch for selecting a configuration source.

## Consequences

Positive:

- Property assignment and clearing match ordinary caller expectations.
- Consumers remain independent of internal composition mechanics.
- Host adapters can present idiomatic APIs while preserving one cross-host meaning.
- Provider defaults remain owned by each provider.
- Requests can explicitly disable output constraints without selecting a configuration source.
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
- Public option representations preserve absence separately from valid false, zero, empty strings, and empty collections through the provider boundary.
- Output constraints distinguish no request choice, an explicitly unconstrained choice, and a complete grammar or schema. Clearing any request choice restores the default; an explicit unconstrained value disables the default constraint.
- A present provider-option list or map replaces the complete value at that option key, including when empty, while unrelated option choices survive.
- The runtime composes caller-owned scopes and performs no settings-file or database I/O.
- Providers own their defaults and receive no separate request and host configuration layers.
- Host adapters translate host-native set and clear operations into runtime intent without exposing runtime composition mechanics.
- Tests cover precedence, assignment, local clearing, provider fallback, explicit false, zero and empty values, whole-value collection replacement, and all three constraint choices.
- New public generation controls describe domain behavior rather than configuration sources or merge operations.

## Notes

- Version: 0.4
- Changelog:
  - 0.4: Define explicit unconstrained intent, preserve presence through provider resolution, clarify option-level replacement, and leave model-loading configuration outside scope
  - 0.3: Narrow the decision to generation configuration, clarify precedence and provider ownership, define collection replacement, and move Godot persistence to ADR-007
  - 0.2: Assign default persistence to hosts, select Godot `ProjectSettings`, and move named profiles to the backlog
  - 0.1: Initial proposed record
