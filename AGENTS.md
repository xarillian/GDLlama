# AGENTS

**Chorus** is a local LLM inference runtime for games initially delivered as a Godot 4.4+ GDExtension backed by `llama.cpp`. This document contains instructions an Agent or AI must follow.

## Structure

Start at `include/chorus/runtime/runtime.hpp` for the public C++ API and
`docs/ARCHITECTURE.md` for the dependency rules. The repository follows those layers:

- `include/chorus/core/`, `src/chorus/core/`: provider-independent domain contracts and their implementations.
- `include/chorus/runtime/`, `src/chorus/runtime/`: host-independent orchestration built only on core contracts.
- `src/chorus/providers/`: concrete inference providers. `echo/` is the dependency-free reference provider; `llama/` owns the `llama.cpp` integration.
- `include/chorus/engine_factory.hpp`, `src/chorus/engine_factory.cpp`: the sole catalog and construction boundary for concrete providers.
- `src/godot_chorus/`, `plugin/`: the Godot host adapter and distributable addon; `include/chorus_c/`, `src/chorus_c/` provide the C host adapter.
- `src/wlib/`: standard-library-only generic utilities.
- `tests/native/`, `tests/godot/`: native layer tests and Godot integration tests. Real-model fixtures live in `tests/models/`.
- `SConstruct`, `justfile`, `tools/`: build graph, common commands, and build helpers. Generated outputs belong in `bin/`; vendored dependencies in `third-party/` are not project source.

## Architecture

@docs/ARCHITECTURE.md

## Comments

Comments should be rare (mythic rare, even). Ensure comments are powerful and describe _why_ rather than _what_. Do not re-produce content from chat context. Do not restate what code is doing. Do not state what can be inferred from domain. Comments must provide new knowledge.

banlist("—", "–", "--"). The ban on "--" is listed if required, such as an argument for a command.

Docstrings should follow the format:

```
/*
 * brief
 *
 * descriptive body
 */
```

or

```
/// brief
```

Concise logs are preferred.

Inline comments get `//`, double slash, regular-comment.

Google-style Python sections come last, after the descriptive body, e.g.:

```
/*
 * Looks up one option in a provider's option schema.
 *
 * The pointer borrows the schema's storage and stays valid
 * until the schema is modified or destroyed; callers must not free it.
 *
 * Returns:
 *  - `const ProviderOptionDescriptor*`: the descriptor whose key matches.
 *  - `nullptr`: the schema declares no such key.
 */
```

Ensure every symbol is backticked and fully qualified, e.g. `ChorusError::Cancelled`, `ChorusRequest::on_event`, `EventType::Stop`. Ensure `Returns:` names what comes back, one bullet per case. `Errors:` should name error vocabulary, and `Raises:` is for code that throws.

### Documentation

Documentation serves its audience, not the implementation history. Keep each document within its purpose. Prefer omission to duplication; code and Git already preserve many details.

## Decisions

Ensure architectural decisions are logged to `docs/ADRs`. This document is tracked and it is _vitally_ important that only high-level decisions are tracked. ADRs can be human or agent written, but they explicitly follow Amazon's ADR style.

The default is no entry. An entry is only earned when a decision binds work that has not happened yet. There are three questions that must be answered "yes" before an ADR is written:

1. Does it constrain future work across a boundary?                
2. Would a reasonable maintainer revisit the decision without its original rationale?                                             
3. Does the decision retain value after the implementation and roadmap entry are gone?    

If the code or a `git log` already store a decision, let the code and `git log` be authoritative. This rule is absolute, the code and `git log` are absolute.

## Git Pratice

Do not include "Authored by <model> ..." in commit messages.

## Testing

This project uses `GoogleTest`.

Tests should not be tautologies. Ensure tests are written where the effect immediately follows the causes; tests should mimic how we speak in natural language.

Not every change needs a unit test, and too many unit tests is a lousy signal. Prefer functional and end-to-end tests. Cover both the happy and the unhappy paths. If a bug would slip past those, an additional unit test is justified. A unit test is also justified for regressions or TDD. Do not blindly add tests. If coverage already exists for a changed piece of code, a test is _very likely_ not necessary.

## Workflow Docs

Specs, plans, process ledgers, and any other agent-workflow files go into a `.gitignored` location. One of:

- `.docs/`
- `_project/`
- `.superpowers/`
- `tmp/`

These documents NEVER enter the tracked `docs/` and are never committed. This rule overrides any skill or tool default: always ensure workflow items are in one of `.docs/`, `_project/`, `.superpowers/`, or `tmp/`.
