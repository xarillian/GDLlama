# ADR-007: Persist generation defaults as portable JSON

Status: Proposed
Date: 2026-09-09

## Context

Generation defaults need a portable representation for standalone hosts and game engines. Godot developers should edit shared defaults through native project settings rather than maintain per-node resources or edit JSON by hand.

Independent defaults in JSON and native settings would create competing values and another precedence rule. Persisting provider defaults would also turn an absent caller choice into a fixed value. We need one authoritative document that stores only caller choices.

## Decision

This decision covers generation defaults. Model paths, loading options, and their persistence remain outside its scope.

- Use JSON (`settings.json`) as the authoritative persisted representation. Native settings surfaces edit that representation rather than supply another defaults layer.
- Support both loading an explicitly selected file and applying supplied JSON contents through the C ABI. Hosts choose which runtime instances receive the defaults; Chorus does not introduce process-wide defaults state.
- Godot imports the JSON when a project opens and presents one shared generation-default set through `ProjectSettings`. Imported choices replace stale stored generation settings. `ProjectSettings` is the preferred editing surface, not an independent source of defaults.
- Host editor changes write back to JSON. Automatic write-back of changed settings is editor-only across all hosts; ordinary runtime changes remain in memory unless explicitly saved. If an editor detects external file changes, it asks the user to reload rather than overwrite them.
- Apply ADR-006's presence and precedence rules: request choices override these host defaults, and absence at both scopes delegates to the provider. Persist only explicit caller choices, never copies of provider defaults.
- When a selected file is missing, create a valid JSON document with no explicit generation choices. Report malformed or unreadable existing files without overwriting them.
- Settings parsing, persistence, and native settings integration belong to host adapters. The runtime receives host-neutral choices as data; core, runtime, and providers perform no settings-file I/O.

## Consequences

Positive:

- Game engines and standalone hosts share one portable settings contract.
- `GodotChorus` nodes share one project-default set without per-node default resources, while C hosts control sharing between runtime instances.
- Native editing remains convenient without creating competing settings sources or freezing provider defaults.

Negative:

- Host editors must synchronize their settings surfaces with JSON and report persistence failures.
- Detected external edits require a reload before editor changes can be saved.

## Compliance

- File and JSON-content inputs produce equivalent caller choices for the selected runtime instances.
- Round trips preserve absence, valid false, zero, empty strings and collections, and explicitly unconstrained choices.
- Missing files become valid empty settings documents without materializing provider defaults; malformed or unreadable files remain unchanged.
- Godot project opening imports the JSON, editor changes write back, and ordinary runtime changes do not automatically modify the file.
- Detected external edits prompt a reload rather than being overwritten.
- Core, runtime, and provider code perform no settings-file or `ProjectSettings` I/O.

## Notes

- Version: 0.2
- Changelog:
  - 0.2: Make JSON authoritative across hosts, support file and content inputs, define editor write-back, and preserve absent choices and existing files on load errors
  - 0.1: Initial proposed draft
