# Examples
This document provides practical examples for using the `GDLlama` node in your projects.

## Basic Text Generation
The most fundamental use is generating text from a prompt. You can do this synchronously, which is blocking, or asynchronously, which is non-blocking.

### Synchronous
```gdscript
func _run_examples() -> void:
    llm.model_path = "res://models/your_model.gguf"
    var error = llm.load_model()
    if error != OK:
        printerr("Failed to load model!")
        return
    
    var prompt = "You are a powerful wizard! Respond with your favorite spell."

    var sync_result = llm.generate_text(prompt)
    print("Result: ", sync_result.strip_edges())

    llm.unload_model()
```

### Asynchronous
```gdscript
# This function is called when the 'generate_text_finished' signal is emitted.
func _on_generation_finished(full_text: String):
    print("[Signal Received] Result: ", full_text.strip_edges())

func _run_examples() -> void:
    llm.model_path = "res://models/your_model.gguf"
    var error = llm.load_model()
    if error != OK:
        printerr("Failed to load model!")
        return
    
    var prompt = "You are a powerful wizard! Respond with your favorite spell."

    # Connect to the signal that fires when the generation is complete.
    llm.generate_text_finished.connect(_on_generation_finished)
    llm.generate_text_async(prompt)

    # We wait here for the signal to be emitted for the example to complete.
    # In a real game, you might not wait and instead let the signal handler do its job.
    await llm.generate_text_finished
    llm.generate_text_finished.disconnect(_on_generation_finished)
```

## Using Chat
In some (many) scenarios, it may be preferable to have the model retain context between turns. To have a conversation where the model remembers previous turns, use `generate_chat` or `generate_chat_async`. These methods maintain context. When you want the model to "forget" the conversation, you can call `reset_context`.

```gdscript
func run_chat_example():    
    # Turn 1: Give the model a secret.
    var response_1 = llm.generate_chat("Remember this secret word: 'avocado'")
    print("NPC Response 1: ", response_1.strip_edges())
    
    # Turn 2: The model should remember the secret from the previous turn.
    var response_2 = llm.generate_chat("What was the secret word?")
    print("NPC Response 2: ", response_2.strip_edges())

    # Now, reset the model's memory.
    print("\n...Resetting context...\n")
    llm.reset_context()
    
    # Turn 3: Ask again. The model has now forgotten the secret.
    var response_3 = llm.generate_chat("What was the secret word?")
    print("NPC Response 3: ", response_3.strip_edges())
```

## Streaming Responses in Real-Time
Often, you'll want to display the text as it's being written. User inference times may be long, and streaming is a way to show that your project is working to the user. This is achieved by connecting to the `generate_text_updated` signal.

```gdscript
@onready var llm: GDLlama = $GDLlama
var full_response: String = ""

func _ready() -> void:
    llm.model_path = "res://models/your_model.gguf"
    
    # Connect to BOTH signals: one for streaming chunks, one for the final result.
    llm.generate_text_updated.connect(_on_text_stream_updated)
    llm.generate_text_finished.connect(_on_text_stream_finished)
    
    var error = llm.load_model()
    if error != OK:
        return

    # Start an async generation. The connected signals will handle the response.
    var prompt = "Tell me a short story about a brave knight."
    llm.generate_text_async(prompt)

# This signal is fired repeatedly with new pieces of text.
func _on_text_stream_updated(new_text_chunk: String):
    # In a real scenario, you would append this new_text_chunk to a Label's text property.
    prints(new_text_chunk) # `prints` prints to the console without a newline.
    full_response += new_text_chunk

# This signal is fired once at the very end.
func _on_text_stream_finished(_full_text: String):
    print("\nGeneration Complete")
```

## Using GBNF Grammar
GBNF (GGML BNF) is a format for defining strict rules for the model's output.

```gdscript
func run_grammar_example():    
    # This grammar attempts to force the model to create a list of specific fruits.
    var fruit_grammar = 'root ::= "A list of fruits:\\n" ("- " ("apple" | "banana" | "orange") "\\n")+'
    var prompt = "List some fruits for me."
    
    print("Grammar: ", fruit_grammar)
    var grammar_result = llm.generate_text(prompt, fruit_grammar)
    print("Grammar Result: ", grammar_result.strip_edges())
```

### Using JSON to Provide Grammar
JSON is schemas are converted to grammar under the hood, but are a reader-friendly way to devise grammar.

```gdscript
func run_json_example():
    var spell_json_schema = """
    {
        "type": "object",
        "properties": {
            "spell_name": { "type": "string" },
            "mana_cost": { "type": "integer" },
            "effect": { "type": "string", "description": "A brief description of the spell's effect." }
        },
        "required": ["spell_name", "mana_cost", "effect"]
    }
    """
    var prompt = "You are a powerful wizard! Invent a spell and describe it."
    
    print("JSON Schema: ", spell_json_schema)
    var json_result = llm.generate_text(prompt, "", spell_json_schema)
    print("JSON Schema Result: ", json_result.strip_edges())
```

## Using Specialized Nodes
You may have cases where you want multiple LLMs in a scene. For this, I recommend multiple nodes.

```gdscript
# Get a reference to both of your specialized LLM nodes
@onready var embedding_llm: GDLlama = $EmbeddingLLM
@onready var chat_llm: GDLlama = $ChatLLM

func _run_all_tests():
    # Load both models
    embedding_llm.load_model()
    chat_llm.load_model()
    
    # Use the embedding model for embeddings
    var s1 = "The cat sat on the mat."
    var emb1 = embedding_llm.compute_embedding(s1)
    
    # Use the chat model for dialogue
    var chat_prompt = "What is your favorite spell?"
    var chat_response = chat_llm.generate_chat(chat_prompt)
    
    print("Chat Response: ", chat_response)
```

## Using Embeddings

```gdscript
var npc_knowledge: Array[String] = [
    "The king's name is Reginald the Bold.",
    "A strange beast has been seen in the Whisperwood Forest to the east.",
    "The best place to get a drink is the Salty Siren Tavern by the docks.",
    "I'm worried about the rising price of iron ore at the market."
]

# Ideally, you'd do the embeddings _before_ the scene, but c'est la vie.
var knowledge_embeddings: Array[PackedFloat32Array] = []

func _ready() -> void:
    # Use call_deferred to ensure the scene tree and nodes are ready.
    call_deferred("_initialize_ai_system")

func _initialize_ai_system() -> void:
    print("Loading AI models...")
    embedding_llm.load_model()
    chat_llm.load_model()
    print("Models loaded.")
    

    # Step 1: Pre-compute embeddings for the NPC's knowledge base.
    print("NPC is recalling its knowledge...")
    for fact in npc_knowledge:
        knowledge_embeddings.append(embedding_llm.compute_embedding(fact))
    print("NPC knowledge is ready.")
    
    # Now, let's ask the NPC a question.
    await _ask_npc("Where can I find a good tavern around here?")

func _ask_npc(player_question: String) -> void:
    print("\nPlayer asks: '%s'" % player_question)
    
    # Step 2: Compute an embedding for the player's question.
    var question_embedding = embedding_llm.compute_embedding(player_question)
    
    # Step 3: Find the most relevant fact using cosine similarity (Semantic Search).
    var best_match_index = -1
    var highest_similarity = -1.0
    
    for i in range(knowledge_embeddings.size()):
        var similarity = embedding_llm.similarity_cos(question_embedding, knowledge_embeddings[i])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_index = i
            
    var relevant_fact = npc_knowledge[best_match_index]
    print("Found relevant fact with similarity %.2f: '%s'" % [highest_similarity, relevant_fact])
    
    # Step 4: Use the chat model to generate a natural response using the fact as context.
    var prompt_template = """
        You are Barnaby, a tired old city guard. Using ONLY the following information, answer the player's question naturally.

        Information: "{fact}"

        Player Question: "{question}"

        Barnaby:
    """

    var final_prompt = prompt_template.format({
        "fact": relevant_fact,
        "question": player_question
    })
    
    print("Generating NPC response...")
    var npc_response = chat_llm.generate_chat(final_prompt)
    print("Barnaby says: '%s'" % npc_response.strip_edges())
```