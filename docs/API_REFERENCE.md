# API
The `GDLlama` node is a custom Godot node that acts as a bridge to the `llama.cpp` library, allowing you to perform complex AI text generation tasks directly from GDScript or C# without halting your game. Key features include:

- Conversational AI: Maintain context between calls to create multi-turn chatbots.
- Non-conversational Responses: One-shot generation for single use text generation.
- Real-time Streaming: Receive text as it's being generated using signals.
- Flexible Generation: Perform both synchronous and asynchronous text generation.
- Model Management: Load and unload .gguf model files at runtime.
- Embeddings: Enable features like semantic search and content similarity checks.

To use it, add a `GDLlama` node to your scene from the "Add Child Node" dialog. You can then give the node any unique name and access it from your scripts like any other node.

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
| `seed` | `int` | `-1` to `4294967295` | The seed for the random number generator. Using the same seed with the same prompt and parameters will produce the exact same output. A value of -1 (default) means a random seed will be used. |
| `n_predict` | `int` | `-1` to context size defined on the model | The maximum number of new tokens the model should generate in a single run. `-1` specifies generation until an EOS token is found or the context is full. |
| `temperature` | `float` | `0.0` to `2.0` | Controls randomness. Higher values (e.g., `1.0`) make the output more random or potentially creative; lower values (e.g., `0.1`) make it more focused and deterministic. |
| `top_k` | `int` | `0` to vocab size | Reduces the pool of tokens to the `k` most likely ones (`0` = disabled). A lower value (e.g., `40`) can prevent strange tokens from appearing. |
| `top_p` | `float` | `0.0` to `1.0` | Nucleus sampling. It considers the smallest set of tokens whose cumulative probability exceeds `p`. A value of `0.9` is a good starting point. |
| `ignore_eos` | `bool` | `true`/`false` | If `true`, the model will not stop when it generates an End-of-Sequence token. |
| `penalty_repeat` | `float` | `1.0` to `2.0` | Penalizes the model for repeating tokens it has recently used (`1.0` = no penalty). |
| `penalty_last_n` | `int` | `0` to context size | The number of recent tokens to consider for the repetition penalty (`0` = disabled). |
| `chat_template` | `string` | N/A | A custom chat template in Jinja2 format. If empty (default), the model's built-in chat template is used automatically. This is an advanced feature for supporting models with non-standard prompt formats or for models with incorrect metadata (or doing weird things). |
| `n_gpu_layers` | `int` | `0` to layer count | The number of model layers to offload to the GPU for acceleration. `-1` uses the default recommended value. |
| `n_ctx` | `int` | `0` to model max | The context window size in tokens. This determines the maximum amount of text (prompt + generation) the model can remember at once. |
| `n_batch` | `int` | `8` to `n_ctx` | The number of tokens to process in parallel during prompt evaluation. A higher value may improve performance on powerful hardware. |
| `main_gpu` | `int` | `0` to number of GPUs | The index of the primary GPU to use for miscellaneous computations when using multi-GPU setups. |

The default values for these properties are provided by `llama.cpp`, as defined by the `common_params_sampling` struct. See: https://github.com/ggml-org/llama.cpp/blob/3d4053f77f0f78ee2b791088c02af653ebee42dd/common/common.h#L137

## Access Methods
| Method | Description |
|---|---|
| `load_model() -> Error` | Loads the model specified by the `model_path` property into memory. Must be called before any generation can occur. **The user is responsible for managing their model in memory!** Returns `@GlobalScope.OK` on success. |
| `unload_model() -> void` | Unloads the currently loaded model from memory, freeing resources. |
| `is_model_loaded() -> bool` | Returns `true` if a model is currently loaded and ready for use, `false` otherwise. |
| `generate_text(prompt: String, grammar: String = "", json: String = "") -> String` | Performs a synchronous (blocking) text generation. This method always starts with a fresh context and applies no text formatting. |
| `generate_text_async(prompt: String, grammar: String = "", json: String = "") -> Error` | Starts an asynchronous (non-blocking) generation. The result is delivered via the `generate_text_finished` signal. Returns `@GlobalScope.OK` on success or `@GlobalScope.FAILED` if another async task is already running. This method always starts with a fresh context and applies no text formatting. |
| `generate_chat(prompt: String, grammar: String = "", json: String = "") -> String` | A synchronous (blocking) method that maintains conversational context between calls. Used for multi-turn conversations. |
| `generate_chat_async(prompt: String, grammar: String = "", json: String = "") -> Error` | The asynchronous (non-blocking) version of generate_chat. Maintains context and delivers the result via a signal. |
| `stop_generate_text() -> void` | Sends a stop signal to the currently running asynchronous generation. The generation will finish its current token and then stop gracefully. |
| `is_running() -> bool` | Returns `true` if an asynchronous generation thread is currently active. |
| `reset_context() -> void` | Clears the model's conversational memory (the KV cache). **The user is responsible for maintaining their context in chat contexts.** |
| `compute_embedding(prompt: String) -> PackedFloat32Array` |  Generates a vector embedding for the given prompt synchronously (blocking). Returns an empty array if embedding is unavailable. |
| `compute_embedding_async(prompt: String) -> Error` | Starts an asynchronous (non-blocking) embedding generation. The result is delivered via the embedding_computed or embedding_failed signal. |
| `similarity_cos(array1: PackedFloat32Array, array2: PackedFloat32Array) -> float` | A utility function that calculates the cosine similarity between two vector embeddings, returning a value between `-1.0` and `1.0`. |

## Signals
| Signal | Arguments | Description |
|---|---|---|
| `generate_text_updated` | `new_text: String` | Emitted repeatedly during an async generation, providing new tokens as they are generated.
| `generate_text_finished` | `full_text: String` | Emitted once when an async generation has completed. Provides the entire text generated during the run.
| `generate_text_error` | `error_text: String` | Emitted once when async generation has completed with an error. |
| `embedding_computed` | `embedding: PackedFloat32Array` | Emitted when an async embedding generation is successful. |
| `embedding_failed` | `error_message: String` | Emitted if an error occurs during async embedding generation. |

## Example Godot Usage
```gdscript
@onready var llm: GDLlama = $GDLlama   # a reference to the GDLlama Node
var full_response: String = ""         # a variable to store the response as it streams in

func _ready():
    # Connect to the signals to receive results from async calls
    llm.generate_text_updated.connect(_on_text_updated)
    llm.generate_text_finished.connect(_on_text_finished)

    # Configure the model path to wherever you have your llama-compatible model stored
    llm.model_path = "res://models/your_model.gguf"

    # Load the model before doing anything else
    if not llm.is_model_loaded():
        var error = llm.load_model()
        if error != OK:
            return
    
    # Start a non-blocking generation
    var prompt = "Tell me a story about the river of Saskatoon."
    llm.generate_text_async(prompt)

    # Here, for example purposes, we wait for the generation to complete.
    # This makes this part of the function "block" while allowing the game to remain responsive.
    # In a real game, you might not await here, and instead let the signal handlers
    # do their work whenever they are called.
    await llm.generate_text_finished

    # Finally, it is best practice to unload the model from memory when you are done with it
    llm.unload_model()

func _on_text_updated(new_text: String):
    # This example function is for STREAMING. It is called repeatedly with new text chunks because
    # we hooked it up to our `generate_text_updated` signal.
    full_response += new_text
    prints(new_text)

func _on_text_finished(full_text: String):
    # This example function is for COMPLETION. It is called once when the entire generation is 
    # finished because we hooked it up to our `generate_text_finished` signal
    print(full_text)
```