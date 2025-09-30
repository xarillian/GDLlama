# Examples
@TODO

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