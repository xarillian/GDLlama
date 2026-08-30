# CLAUDE.md

## What This Is
A Godot 4.4+ GDExtension that wraps `llama.cpp` to provide local LLM inference in games. Built with SCons, targeting Windows/Linux/macOS. The project's thesis is "Real-time inference at scale on one local GPU". We should be able to run 40+ NPC conversations at one time on consumer hardware.

Ongoing work currently lives in `_project`

## Architecture
@docs/ARCHITECTURE.md

## Tests
Custom lightweight test framework in `tests/native/support/test_utils.hpp` (macros: `ASSERT_TRUE`, `ASSERT_EQ`, color output), plus the shared fake `tests/native/support/sync_mock_engine.hpp` (`SyncMockEngine`: deterministic, inline-emitting, contract-conforming).

The native tree mirrors the layers:

- `tests/native/core/` — contract/value-type unit tests (no engine, no model)
- `tests/native/runtime/` — request-lifecycle, sessions, chat history, prompt fitting
- `tests/native/factory/` — provider factory tests
- `tests/native/wlib/` — utility tests (UTF-8 handling)
- `tests/native/providers/echo/` — reference provider unit tests
- `tests/native/providers/llama/` — suites hitting real llama.cpp (require the model)
- `tests/native/test_runner.cpp` — `main()` entry point

GDScript integration tests live in `tests/godot` (staged by `just godot`).

## Workflow Docs
Specs, plans, progress ledgers, and any other agent-workflow files go into a gitignored location. One of: `.docs/` or or `_project` or `.superpowers/`. NEVER in tracked `docs/` and NEVER committed. This overrides any skill or tool default (e.g. `docs/superpowers/...`). 

Do NOT include "Authored by..." in commit messages.

### Feature Set

@_project/FEATURE.md

### Decisions

Provide decision updates to the ADR-like doc a DECISIONS.md, stored at `_project/DECISIONS.md`. This is currently only stored locally and is NOT pushed.

The default is no entry. One is earned only when the decision binds work that
has not happened yet: a contract another layer must program against, a direction
that closes off alternatives, an argument that would otherwise be had again. If
the code and `git log` already carry it, they are the record. Deleting dead
code, renaming, fixing a bug, and reversing an entry whose subject no longer
exists are not decisions. The entries already in the file are the calibration.
When in doubt, propose the entry to me in chat instead of writing it.

### Comments

Docstrings should follow the format:

```
/*
 * brief
 *
 * descriptive body
 */
```

```
/// brief
```

is also completely acceptable. Not all logs need to be verbose; use judgment.

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

- One line per bullet. Type, value, or symbol first, colon, then the condition.
  Anything needing a sentence belongs in the body.
- Backtick and fully qualify every symbol: `ChorusError::Cancelled`,
  `ChorusRequest::on_event`, `EventType::Stop`. Never the bare leaf name.
- `Returns:` names what comes back, one bullet per case.
- `Errors:` names the error vocabulary. `Raises:` is for code that throws.
- Body prose says how failure travels: returned, or signalled on
  `ChorusRequest::on_event` from an engine thread.
- Sections are earned. Write one when the return has cases or the failure has a
  vocabulary. Never write a section to say "none".

## Key Build Notes

- C++20 required
- Commands (via `just`):
  - `just build [--cpu]` — shared library; Vulkan by default, `--cpu` for a plain CPU build
  - `just release [--cpu]` — same, `target=template_release`
  - `just check [--quick] [filter]` — build the test binary, then run the suite; the full run is the gate and requires the model at `tests/models/gemma-3-270m-it-F16.gguf`
  - `just test [--quick] [filter]` — run the suite without rebuilding
  - `--quick` on either skips model suites (`CHORUS_SKIP_MODEL_TESTS=1`); `filter` is a test-name substring, e.g. `just check --quick Echo`
  - `just godot` — build, then stage the plugin into `plugin/addons/chorus` and symlink it into `tests/godot`
  - `just compiledb` — regenerate `compile_commands.json` for clangd
- Links against llama.cpp static libs: `llama-common`, `llama-common-base`, `llama`, `ggml`, `ggml-cpu`, `ggml-base`
- macOS needs Metal/MetalKit/Foundation/Accelerate frameworks
- Linux links OpenMP
- Output: `bin/libgodot_chorus` (shared lib) + `bin/run_tests` (test binary)
