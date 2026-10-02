# ADR-000: Rewrite GDLlama as Chorus, a host- and provider-independent runtime

Status: Proposed
Date: 2026-07-01

## Context

GDLlama 1.0 was a Godot node that drove `llama.cpp` directly. Each `GDLlama` node ran one generation at a time on one thread, decoded a single sequence, and reported results through signals that carried no request identity. Game AI, where many agents answer one event, needed batch processing (issue #59). With 1.0, a game could load a model per agent or queue every request behind the last.

Batching could not be added in place. It needed request identity, ordering, per-request delivery and thread safety (issues #67, #70 and #74), and every one of those ran through one node class that also applied chat templates and called `llama.cpp` itself (issues #37 and #62). That class bound each feature to both Godot and `llama.cpp`, so supporting another engine or inference library (issue #61) would have meant building it all again.

## Decision

Rewrite the project top-down as Chorus instead of extending 1.0:

- Build concurrency, batching, thread safety, and priority scheduling into the runtime from the start.
- Separate a host-independent runtime from host adapters, with Godot first, and from inference providers, with `llama.cpp` first. Another engine or library becomes an adapter or a provider, not a fork. ADR-003 records the dependency rules.
- Give the project its own identity: GDLlama becomes Chorus.
- Replace the 1.0 Godot API without a compatibility layer. Version 2.0 documents the migration.

## Consequences

Positive:

- Many agents share one loaded model through shared batches, with request identities, priorities, cancellation, and one terminal result per accepted request.
- Hosts and providers are replaceable; the C++ runtime and C ABI already serve hosts other than Godot.
- Core and runtime behavior can be tested without a model, vendor library, or running host.

Negative:

- Projects built on 1.0 must migrate their scenes and scripts by hand.
- A feature now crosses core, runtime, provider, and host layers, which costs more code than one node class did.

## Compliance

- New hosts and providers integrate through the runtime and provider factory boundaries in `docs/ARCHITECTURE.md`; none reimplements the runtime.
- Work that needs concurrency, request identity, or ordering extends runtime contracts, not a host adapter.
- Public names use Chorus.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial record, written after the decision from PR #76 and issues #37, #59, #61, #62, #67, #70 and #74.
