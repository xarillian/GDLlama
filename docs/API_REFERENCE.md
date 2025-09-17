# API
The `GDLlama` node is a new node in Godot that acts as a bridge to the `llama.cpp` library, allowing you to perform complex AI text generation tasks directly from GDScript without halting your game.

To use it, add a GDLlama node to your scene from the "Add Child Node" dialog. You can then give the node any unique name and access it from your scripts like any other node.

Getting started:
1) Add a `GDLlama` node to your scene.
2) Set the `model_path` property to your desired, llama-compatible model.
3) Reference the node and call `load_model` to load the model into memory.
4) You can now successfully call generation methods like `generate_chat_async` and connect to its signals to receive the results.
5) Unload the model. This will happen automatically when the instance is destroyed, but it is good practice to do so.

## Properties
These properties belong to a `GDLlama` node and can be set via code or the Godot Editor's inspector.

| Property | Type | Range | Description |
|---|---|---|---|
| `model_path`| `string` | N/A | The file path to your llama-compatible model file (e.g., `"res://models/model.gguf"`) |
| `n_predict` | `int` | `-1` to context size defined on the model | The maximum number of new tokens the model should generate in a single run. `-1` specifies generation until an EOS token is found or the context is full. |
| `temperature` | `float` | `0.0` to `2.0` | Controls randomness. Higher values (e.g., `1.0`) make the output more random or potentially creative; lower values (e.g., `0.1`) make it more focused and deterministic. |
| `top_k` | `int` | `0` to vocab size | Reduces the pool of tokens to the `k` most likely ones (`0` = disabled). A lower value (e.g., `40`) can prevent strange tokens from appearing. |
| `top_p` | `float` | `0.0` to `1.0` | Nucleous sampling. It considers the smallest set of tokens whose cumulative probability exceeds `p`. A value of `0.9` is a good starting point. |
| `ignore_eos` | `bool` | `true`/`false` | If `true`, the model will not stop when it generates an End-of-Sequence token. |
| `penalty_repeat` | `float` | `1.0` to `2.0` | Penalizes the model for repeating tokens it has recently used (`1.0` = no penalty). |
| `penalty_last_n` | `int` | `0` to context size | The number of recent tokens to consider for the repetition penalty (`0` = disabled). |

The default values for these properties are provided by `llama.cpp`, as defined by the `common_params_sampling` struct. See: https://github.com/ggml-org/llama.cpp/blob/3d4053f77f0f78ee2b791088c02af653ebee42dd/common/common.h#L137

## Access Methods
| Method | Description |
|---|---|
| `load_model() -> Error` | Loads the model specified by the `model_path` property into memory. Must be called before any generation can occurs. **The user is responsible for managing their model in memory!** Returns `@GlobalScope.OK` on success. |
| `unload_model() -> void` | Unloads the currently loaded model from memory, freeing resources. |
| `is_model_loaded() -> bool` | Returns `true` if a model is currently loaded and ready for use, `false` otherwise. |
| `generate_text(prompt: String, grammar: String = "", json: String = "") -> String` | Performs a synchronous (blocking) text generation. This method always starts with a fresh context. |
| `generate_text_async(prompt: String, grammar: String = "", json: String = "") -> Error` | Starts an asynchronous (non-blocking) generation. The result is delivered via the `generate_text_finished` signal. Returns `@GlobalScope.OK` on success or `@GlobalScope.FAILED` if another async task is already running. |
| `generate_chat(prompt: String, grammar: String = "", json: String = "") -> String` | A synchronous (blocking) method that maintains conversational context between calls. Used for multi-turn conversations. |
| `generate_chat_async(prompt: String, grammar: String = "", json: String = "") -> Error` | The asynchronous (non-blocking) version of generate_chat. Maintains context and delivers the result via a signal. |
| `stop_generate_text() -> void` | Sends a stop signal to the currently running asynchronous generation. The generation will finish its current token and then stop gracefully. |
| `is_running() -> bool` | Returns `true` if an asynchronous generation thread is currently active. |
| `reset_context() -> void` | Clears the model's conversational memory (the KV cache). **The user is responsible for maintaining their context in chat contexts.** |

## Signals
| Signal | Arguments | Description |
|---|---|---|
| `generate_text_updated` | `new_text: String` | Emitted repeatedly during an async generation, providing new tokens as they are generated.
| `generate_text_finished` | `full_text: String` | Emitted once when an async generation has completed. Provides the entire text generated during the run.
| `generate_text_error` | `error_text: String` | Emitted once when async generation has completed with an error. |

## Example Godot Usage
```
@onready var llm: GDLlama = $GDLlama   # a reference to the GDLlama Node

func _ready():
    # Connect to the signals to receive results from async calls
    llm.generate_text_updated.connect(_on_text_updated)
    llm.generate_text_finished.connect(_on_text_finished)

    # Load the model before doing anything else
    if not llm.is_model_loaded():
        var error = llm.load_model()
        if error != OK:
            return
    
    # Start a non-blocking generation
    var prompt = "Tell me a story about the river of Saskatoon."
    llm.generate_text_async(prompt)

func _on_text_updated(text: String):
    prints(text)

func _on_text_finished(text: String):
    print(text)
```