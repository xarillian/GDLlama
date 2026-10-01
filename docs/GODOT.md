# Godot

## Compatibility

| Godot Version | Compatible |
| --- | --- |
| 4.4 | ✓ |
| 4.4.1 | ✓ |
| 4.5 | ✓ |
| 4.5.1 | ✓ |
| 4.5.2 | ✓ |
| 4.6 | ✓ |
| 4.6.1 | ✓ |
| 4.6.2 | ✓ |
| 4.6.3 | ✓ |
| 4.7 | ✓ |
| 4.7.1 | ✓ |
| 4.7.2 | ✓ |

## Installation

### Compilation

You need Git, Python 3 with SCons, CMake, and a C++20 compiler. Run the following commands from your clone of the Chorus repository.

Initialize the dependencies:

```sh
git submodule update --init --recursive
```

Choose a build:

| Build | Command | Additional requirement |
| --- | --- | --- |
| Vulkan, Linux or Windows | `scons use_vulkan=yes` | Vulkan SDK and a compatible GPU driver |
| Metal, macOS | `scons use_metal=yes` | Xcode |

Then prepare the addon for copying into your project:

```sh
python tools/stage_godot.py
```

The resulting addon is in `plugin/addons/chorus`, including `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `licenses/`. Keep these notices when redistributing the addon. For exported games, ship them alongside the executable or explicitly include them in the export; do not assume Godot exports non-resource files.

### Plugin installation

1. Copy `plugin/addons/chorus` into your project's `addons` directory.
2. Open the project and let Godot finish importing it.
3. Open _Project > Project Settings > Plugins_ and enable **Chorus LLM**.

## Basic usage

### Model selection

Chat generation requires a GGUF model with a chat template. `Qwen3-0.6B-Q8_0.gguf` is one small instruction-tuned option. The model path can be an absolute filesystem path or a Godot `res://` or `user://` path. Chorus does not include or download a model.

### Scene configuration

A `GodotChorus` node handles model loading and generation. The example below accesses it as a child named `GodotChorus`; adapt the node path to your scene.

Example Inspector configuration for GPU inference:

| Property | Value |
| --- | --- |
| Provider | Llama |
| Model Path | Path to your GGUF file |
| Use GPU | Enabled |
| GPU Layers | `-1` (all layers) |
| Context Size | `4096` |
| Max Concurrent Requests | `1` |

### Text generation

This example loads the model and submits one chat request, handling both immediate rejection and asynchronous failure:

```gdscript
extends Node

@onready var chorus: GodotChorus = $GodotChorus

var load_id: int = -1
var request_id: int = -1


func _ready() -> void:
    chorus.model_loaded.connect(_on_model_loaded)
    chorus.model_load_failed.connect(_on_model_load_failed)
    chorus.generation_complete.connect(_on_generation_complete)
    chorus.generation_error.connect(_on_generation_error)

    var result := chorus.load_model()
    if not result.accepted:
        push_error("Model load rejected: " + result.message)
        return
    load_id = result.load_id
    print("Loading model...")


func _on_model_loaded(id: int, _model_id: String) -> void:
    if id != load_id:
        return
    load_id = -1
    print("Model ready.")

    var request := ChorusRequest.chat(&"guide", "Greet a traveler in one sentence.")
    request.max_tokens = 64
    request.show_thinking = false
    var result := chorus.generate(request)
    if not result.accepted:
        push_error("Generation rejected: " + result.message)
        return
    request_id = result.request_id


func _on_model_load_failed(id: int, _model_id: String, code: int, message: String) -> void:
    if id != load_id:
        return
    load_id = -1
    push_error("Model load failed (%d): %s" % [code, message])


func _on_generation_complete(id: int, _session: StringName, _message_id: int,
        content: String, _reasoning: String) -> void:
    if id != request_id:
        return
    request_id = -1
    print("Reply: " + content)


func _on_generation_error(id: int, _session: StringName, code: int, message: String) -> void:
    if id != request_id:
        return
    request_id = -1
    push_error("Generation failed (%d): %s" % [code, message])
```

On success, the example prints `Loading model...`, `Model ready.`, and `Reply:` followed by generated text.

An accepted load is still in progress: wait for `GodotChorus.model_loaded` before requesting a reply. An accepted generation can also fail later, so handle both the immediate result and the error signal, as the example does.

Keep the node in the scene tree and processing while waiting for results.

## Default generation settings

For the Llama provider, unset sampling settings such as temperature, `top_k`, and `top_p` use the defaults from the bundled version of llama.cpp. Chorus does not automatically apply the model author's recommended generation settings!

Find `chorus/generation` in Project Settings. The enabled plugin saves valid changes to `res://chorus/settings.json`; keep that file in version control.

For common choices, an empty Dictionary means no project override. A Dictionary with a typed `value` entry selects a value, such as `{"value": 128}` for `max_tokens`. A selected zero or `false` is a real choice, not an unset field.

Request choices override project choices, which override provider defaults. The example's `request.max_tokens = 64` takes precedence over the project setting. Remove that assignment, or call `request.clear_max_tokens()`, to use the default instead.