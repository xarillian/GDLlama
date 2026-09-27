# Chorus
**Chorus** is a local LLM inference runtime for games initially delivered as a Godot 4.4+ GDExtension backed by `llama.cpp`.

The project’s thesis is that one consumer GPU should support real-time inference for many concurrent game agents. Its core is independent of any host or inference provider, so `Godot` and `llama.cpp` are the first integrations rather than permanent boundaries.      

Chorus is under active development; it is currently the hobby project of one dev. The native runtime, `llama.cpp` provider and a Godot adapter are implemented. 

## Features

## Setup
Chorus requires Git, Python 3 with SCons, CMake 3.14 or newer, a C++20 compiler toolchain, and [Just](https://github.com/casey/just). The default Linux and Windows build also requires the Vulkan SDK.

After cloning Chorus, run:

```sh
git submodule update --init --recursive
just build
just check --quick
```

On macOS, install Xcode and use the Metal build in place of `just build`:

```sh
scons use_metal=yes
```

Build artifacts are written to `bin/`. The full `just check` suite also runs real-model tests and expects `tests/models/gemma-3-270m-it-F16.gguf`.

The pinned llama.cpp source is kept clean. `SConstruct` reconstructs it in `bin/vendor/llama.cpp/` and applies `patches/llama-resource-cleanup.patch` before compiling either backend or the provider. The patch closes the non-mmap asynchronous upload cleanup gap on cancellation, partial setup, and exceptions; it also releases a partially constructed compatibility batch when conversion allocation fails and preserves the correct GGML revision when CMake runs beneath the Chorus checkout. The source tree and CMake build identity include the vendor revision and patch/helper bytes. A patch conflict or dirty tracked submodule fails the build rather than using unpatched code. `.gitattributes` keeps the patch LF even in CRLF-configured checkouts, matching Git's pristine vendor archive. For an upstream revision change, check out the public revision in `third-party/llama.cpp`, rebase the three-file patch against it, and rerun both CPU and Vulkan build/tests plus the upload probe. Do not commit a local-only vendor revision. The weekly latest-llama canary intentionally fails if the patch drifts. On a Linux Vulkan machine with the existing GGUF fixture, `scons test use_vulkan=yes -j2` followed by `python tools/run_vendor_upload_probe.py` checks repeated upload cancellation, final cancellation, decode, bounded partial-setup failures, and conversion allocation faults against the exact built archive. The probe uses non-mmap loading only within the test; Chorus production defaults are unchanged.

### Godot

To build and stage the addon for Godot 4.4 or newer on Linux or Windows, run:

```sh
just godot
```

After a Metal build on macOS, stage the addon without rebuilding:

```sh
python tools/stage_godot.py
```

The staged addon is written to `plugin/addons/chorus`. Its declared minimum remains Godot 4.4. Actual editor and runtime acceptance passed on Godot 4.4.1. The installed Arch Godot 4.7.2 aborts a fresh editor import with the current godot-cpp 4.4.1 bindings; a matched minimal extension registering one native class aborts at the same phase. Compatibility with that engine version remains unresolved.

To use it in a Godot project:

1. Copy that directory to `<project>/addons/chorus`.
2. Open the project. The extension imports `res://chorus/settings.json` even before the editor plugin is enabled; if missing, it creates `{"version":1,"generation":{}}`. Keep this file **outside** `addons/chorus`, which staging replaces.
3. Enable **Chorus LLM** under **Project Settings > Plugins** for editor write-back and reload prompts.
4. Add a `GodotChorus` node to a scene and set its `model_path` to a compatible GGUF model.
5. Connect load signals, call `load_model()`, and check its typed `ChorusLoadResult.accepted` before waiting for the identified terminal.

```gdscript
var terminals := {}
chorus.model_loaded.connect(func(id, model_id): terminals[id] = true)
chorus.model_load_failed.connect(func(id, model_id, error_code, message):
    terminals[id] = false
    if error_code == GodotChorus.ERR_CANCELLED:
        print("Model load %d cancelled" % id)
    else:
        push_error("Model load %d failed (%d): %s" % [id, error_code, message]))
chorus.model_load_progress.connect(func(id, model_id, phase, has_fraction, fraction):
    if has_fraction:
        print("Load %d phase %d: %.1f%%" % [id, phase, fraction * 100.0])
    else:
        print("Load %d phase %d: progress unknown" % [id, phase]))
var load := chorus.load_model()
if not load.accepted:
    push_error("Load rejected: %s" % load.message)
else:
    # To abandon this attempt, call chorus.cancel_load(load.load_id) and still await its terminal.
    var deadline := Time.get_ticks_msec() + 30000
    while not terminals.has(load.load_id) and Time.get_ticks_msec() < deadline:
        await get_tree().process_frame
    if terminals.get(load.load_id, false) and chorus.is_loaded():
        var result := chorus.generate(ChorusRequest.stateless("Hello"))
        assert(result.accepted)
    elif not terminals.has(load.load_id):
        push_error("Timed out waiting for load %d" % load.load_id)
```

A weight-loading fraction of 100% is not readiness. A success signal records publication, but an earlier handler in the same polled batch can stop or replace the engine; correlate the ID and check current readiness when acting. Changing node properties while loaded or loading affects only the next accepted load. Cancellation admission does not wait for cleanup; explicit `stop_all()` and node destruction do. `model_path` accepts absolute paths and Godot `res://` or `user://` paths to loose GGUF files. A model packed inside a PCK must currently be extracted to the filesystem.

#### Shared generation defaults

`settings.json` is the sole persisted authority for **generation choices**, not model paths or load settings. The default source is `res://chorus/settings.json`; `chorus/generation/settings_path` selects another project-contained `res://` JSON file. Bundle that selected file in **Project > Export > Resources** (for example, include `*.json` or the selected path in non-resource export filters), and check the exported PCK contains it. A missing selected file in a read-only export cannot be created: the host reports an error and uses empty caller choices rather than stale `project.godot` values. Existing malformed or unreadable JSON is not overwritten. Godot rejects JSON text containing an escaped NUL in a string or option key without changing the file, since its native String boundary cannot represent it losslessly; the portable codec and C API retain escaped NUL. On project opening, imported JSON replaces cached `chorus/generation/` choices in `project.godot`; failed first import clears stale choices and reports a diagnostic in the editor output or game log. A failed later reload keeps the last valid choices and source. Fix the file and reload explicitly instead of treating cached settings as a fallback.

The portable document has exactly `version: 1` and an object `generation`. Omitted choices mean the provider decides if neither a request nor the shared defaults supplies a value. A present request choice replaces the corresponding project choice; clearing the request choice resumes project/provider resolution. The following file illustrates valid presence, including zero, false, an empty collection and an explicit unconstrained output:

```json
{"version":1,"generation":{"max_tokens":0,"show_thinking":false,"stop":[],"seed":18446744073709551615,"constraint":{"kind":"unconstrained"},"provider_options":{"llama":{"repeat_penalty":1.0,"logit_bias":{}}}}}
```

Supported `generation` keys are `max_tokens` and `top_k` (signed 32-bit integers), `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` (finite 32-bit floats), `seed` (unsigned 64-bit integer), `stop` (string array), `show_thinking` (boolean), `chat_template` (string), `constraint`, and `provider_options`. A constraint is either `{"kind":"unconstrained"}` or an object with `kind` set to `gbnf`, `json_schema`, `regex`, or `lark` and a string `source`. `provider_options` maps provider names to option dictionaries: option values are booleans, signed 64-bit integers, finite floating numbers, strings, arrays, or string-keyed dictionaries. A present option map, even `{}`, replaces that entire request/default option value rather than recursively merging. Unknown schema fields, `null`, duplicate keys and malformed data produce errors, not silently dropped choices. A syntactically valid choice unsupported by the selected provider can still fail when a request is prepared.

Use **Project Settings > General > chorus/generation** to edit shared choices. Each common choice and `chat_template` is a Dictionary: `{}` means absent. Add a key named `value` with the desired typed value to select it (including `0`, `false`, `""`, `[]`); edit that value to replace it, or delete the key to clear it. `constraint` uses a Dictionary as its selected `value`; `{"value":{"kind":"unconstrained"}}` differs from absent `{}`. Enter `seed` as **decimal text** in the selected `value` field so its full unsigned range is retained. `provider_options` is a direct Dictionary of namespaces and keys. Editor edits write back after a short delay; if the file changed outside Godot, the plugin asks **Reload** (discard unsaved edits) or **Cancel** (keep them in memory without overwriting disk). Invalid edits and failed writes appear as editor diagnostics. Gameplay edits to ProjectSettings affect later requests in memory only; `ChorusProjectSettings.new().save_generation_defaults()` explicitly saves after the same source/conflict checks, returning a Dictionary with `status` and `message`. `reload_generation_defaults()` reimports the selected file; failed sources block saves until a successful reload. Accepted requests retain their admission snapshot.

Scenes with older `GodotChorus.generation_defaults`, node `chat_template`, or `_generation_choices` data must move their **explicit** generation values to the shared `settings.json`; remove those obsolete node properties/resources from scenes. `ChorusRequest.chat_template` and its other request-local properties remain available. Do not migrate old `project.godot` cache entries as defaults, or copy provider-advertised default values into the file. Multiple nodes in a project share one set of choices; for different agent behavior, put concrete choices on each request.

#### C hosts

The C ABI (version 8) has no process-wide defaults or implicit write-back. For each selected `chorus_runtime*`, call `chorus_generation_defaults_load_file(rt, path)` or `chorus_generation_defaults_apply_json(rt, bytes, byte_count)`; these replace that runtime's host choices on success only. Load creates a valid empty document if the selected path is missing; passing empty JSON content is an error. `chorus_generation_defaults_export_json(rt, &json)` returns a caller-owned UTF-8 string (free it with `chorus_string_free`). `chorus_generation_defaults_save_file(rt, path)` writes to an **explicitly selected destination**, including defaults last applied as JSON content. No load/apply remembers a save target; reload requires another load call. Existing malformed or unreadable destinations are not replaced. Invalid arguments/JSON return `CHORUS_ERR_INVALID_REQUEST`; I/O/allocation failures return `CHORUS_ERR_UNKNOWN`. Inspect `chorus_last_error_message(rt)` for the scoped diagnostic, which clears on success. Request choices override that runtime's injected choices and clearing only removes the request choice.

## Architecture
See [Architecture](docs/ARCHITECTURE.md) for the dependency model, service
contracts, provider boundaries, and threading invariants.

## License
Licensing terms are still being determined.
