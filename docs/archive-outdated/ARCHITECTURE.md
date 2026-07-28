# Architecture Patterns
There are two primary architecture patterns I'll identify for use with `GDLlama`: the Worker Node and the Singleton pattern. The best choice depends on your specific needs, expected user resources, and your model choice. A hybrid architecture is also possible. 

## Worker Node Pattern
This is the simplest and most direct way to use `GDLlama`. In this pattern, a specific node that requires generative capabilities (an NPC, maybe) has its own `GDLlama` node as a direct child. This forms a self-contained unit.

Your scene tree may look like:
```
- NPC
  - GDLlama
  - CharacterBody2D
  - ...
```

You would then attach the script to the `NPC` node. It manages its own child `GDLlama` instance.

```GDScript
extends CharacterBody2D

# A direct reference to the `GDLlama` instance.
@onready var llm: GDLlama = $GDLlama

func _ready():
    # This NPC manages its own model loading and signals.
    llm.model_path = "res://models/your_model.gguf"
    llm.generate_text_finished.connect(_on_finished)
    
    var error = await llm.load_model()
    if error != OK:
        printerr("NPC failed to load its model!")

func talk_about(topic: String):
    if not llm.is_model_loaded() or llm.is_running():
        return # Not ready or already telling a story.

    var prompt = "Talk about %s." % topic
    llm.generate_chat_async(prompt) # Use chat for conversational memory in this node.

func _on_finished(full_text: String):
    print("NPC: '", full_text.strip_edges(), "'")
    # Here you would display the text in a dialogue bubble, play audio, etc.

func _exit_tree():
    # Clean up when the NPC is removed from the scene.
    llm.unload_model()
```

## Singleton (Global Service) Pattern
A more memory-efficient pattern for model usage. You have a single, globally accessible `GDLlama` instance that manages a single loaded model. Any node in the game can send requests to this service, which processes them one-by-one in a queue. This pattern requires significantly more diligent context management than the Worker Node Pattern. If there is any point you would take from these docs, and you intend to use this pattern, let it be the need for context management.

**Example**
```GDScript
extends Node

@onready var llm: GDLlama = $GDLlama

var is_busy: bool = false
var request_queue: Array = []

# A struct-like class to hold request data.
class LlamaRequest:
    var prompt: String
    var caller: Callable # The function to call with the result.

    func _init(p_prompt: String, p_caller: Callable):
        self.prompt = p_prompt
        self.caller = p_caller

func _ready():
    # The service connects to its own worker's signals.
    llm.generate_text_finished.connect(_on_generation_finished)
    llm.model_path = "res://models/your_model.gguf"
    
    var error = await llm.load_model()
    if error != OK:
        printerr("CRITICAL: LlamaService failed to load model!")
        get_tree().quit()

# A public function that any node in the game can call.
func request_generation(prompt: String, caller_callback: Callable):
    var new_request = LlamaRequest.new(prompt, caller_callback)
    
    if not is_busy:
        _process_request(new_request)
    else:
        print("LLM is busy. Adding request to queue.")
        request_queue.push_back(new_request)

func _process_request(request: LlamaRequest):
    is_busy = true
    # This example uses generate_text for simplicity, which always uses a fresh context.
    # It's likely the bulk of your work would be done in this method or one like it!
    llm.generate_text_async(request.prompt)

func _on_generation_finished(full_text: String):
    # Here, you would identify where the original request came from and call their callback function with the result.
    # This example isn't great. It skips all that logic and goes to `process_request`.

    ...

    if not request_queue.is_empty():
        var next_request = request_queue.pop_front()
        _process_request(next_request)
```

When you actually would like to use your service, you would do something like:
```GDScript
extends CharacterBody2D

func generate_quest_dialogue():
    var prompt = "Create a short, urgent quest objective for the player."
    
    LlamaService.request_generation(prompt, Callable(self, "on_dialogue_received"))
    print("QuestGiver sent a request to the LlamaService.")

func on_dialogue_received(text: String):
    print("QuestGiver received dialogue: ", text)
```

### Event Bus
You may want to decouple your service, also. A global event bus can solve this problem. You'd instead call a general-purpose signal, which is picked up on by the service. Your game nodes don't need to know anything about the LLM service in this case, which may be quite handy. I haven't prototyped this at all, so I am not entirely sure of its utility.

### Other Upgrades?
There are many upgraded forms of the Singleton Service. Utilizing a priority queue, creating some kind of context pool manager, etc. etc. If you have a usage that isn't supported, post [an issue to the GitHub](https://github.com/xarillian/GDLlama/issues) or create a PR.

# Databases and Vector Storage
`GDLlama` provides the core tools to create vector embeddings and calculate similarity between them, but it does not include a built-in vector database for storage or advanced querying. You are responsible for managing how and where you store your generated embeddings. This may be a point of future development, but for now remains true.

The ideal storage solution depends entirely on the scale and requirements of your project.

## In-Memory Storage
For many games, especially those with a limited amount of searchable data, or in contexts where long-term storage doesn't matter as much, a simple in-memory approach is perfectly fine. You can simply store embeddings as a `PackedFloat32Array` array, which is simple and fast. An example is provided in [EXAMPLES.md](EXAMPLES.md#Using_Embeddings)

## File-Based Storage
If you need to persist your embeddings or prompts between game sessions but don't require a full database server, saving them to a file is a great option.

### Using Godot's `FileAccess` Object
### Using JSON Files

## External Database
The best approach is to use a dedicated vector database that runs as an external service. Your Godot application could communicate with this database over the network using the `HTTPRequest` node.

A database can also be added to Godot through the GDExtensions feature.

# Agentic Tool Use (MCP-Inspired Architecture)
A powerful architecture you can build with `GDLlama` is a tool-use pattern. This turns your LLM from a simple text generator into an active agent that can interact with your game world. The core loop is simple: the LLM generates a structured function call, your code executes that function, and the result is fed back to the LLM to inform its next action.
`GDLlama` is well suited for this pattern, as it provides the two most critical components:
1. Conversational Context: The `generate_chat_async` method remembers the history of the conversation, which is essential for stateful, multi-step actions.
2. Structured Output: The `json` parameter in the generation methods allows you to force the model's output to conform to a specific JSON schema, guaranteeing a predictable, machine-readable tool call.

Future updates to `GDLlama` will further support the pattern. 

## MCP Architecture
It's important to distinguish this internal pattern from the formal Model Context Protocol (MCP) specification. The pattern described here is a self-contained loop inside your Godot application. The formal MCP specification is a standardized protocol for communication between separate services (e.g., an LLM service, a game client, and a tool server communicating over a network).
For a self-contained or monolithic system, implementing the full, formal protocol is likely unnecessary. However, the specification provides a valuable philosophical pattern and an excellent reference for how to think about structuring tool definitions and handling agent-based logic. 
For a project that relies on external communication, the formal MCP specification becomes a highly practical guide. Adhering to its standards will ensure your system is robust, scalable, and interoperable with other tools that speak the same protocol.
For more information on the MCP pattern, see: https://modelcontextprotocol.io/docs/learn/architecture