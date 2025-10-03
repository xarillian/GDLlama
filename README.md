# GDLlama
> Isn't it cool to utilize large language model (LLM) to generate contents for your game?
- @Adriankhl, original creator of GDLlama

Why, yes, I do think it is cool! 

`GDLlama` is a GDExtension for Godot 4.2+ that acts as a bridge to the powerful `llama.cpp` library. This allows you to perform fast, local inference with Large Language Models (LLMs) directly in your game, without needing an internet connection or external servers.

It's implemented as a custom GDLlama node that you can add to any scene, making generative AI a native part of your project. It's designed to be flexible and powerful, supporting key features like:
- Conversational AI: Maintain context between calls to create multi-turn chatbots.
- Non-conversational Responses: One-shot generation for single use text generation.
- Real-time Streaming: Receive text as it's being generated using signals.
- Flexible Generation: Perform both synchronous and asynchronous text generation.
- Embeddings: Enable features like semantic search and content similarity checks.

The generative space is an exciting frontier for video games that has been sorely under-explored so far. LLMs and multimodal models have a great potential to complement multiple aspects of game design, from dialogue generation to quest generation and beyond. Thanks to `llama.cpp`, we can perform inference fast enough locally to enable some genuinely interesting gameplay. I want to help Godot at least keep pace with Unity and Unreal.

I intend to maintain this for an indefinite amount of time while it continues to be useful to me. This is a fork of [Adriankhl's original godot-llm](https://github.com/Adriankhl/godot-llm) with updated build instructions and fixes for recent `llama.cpp` versions. It has been almost entirely re-written with a number of new features. For a full release, I intend to publish this to the Godot Asset Shop. For progress, see: https://github.com/xarillian/GDLlama/milestone/1

# Getting Started
For now, everything has to be built by the user. GDLlama is not yet in the asset library, no sir. I'd like a bit more polish on the project before getting to a 1.0 release state where I'd put it in the library.

## Build
You'll need these tools:
- CMake 3.14+
- Ninja build system
- Vulkan SDK (for GPU builds)
- Git
- (for Windows): Visual Studio Build Tools with clang-cl
    - or some equivalent

Then see the build steps: [docs/BUILD.md](docs/BUILD.md)

## API
There are three main access methods the moment:
- `load_model` -> Used to load the model into memory.
- `generate_text_async` -> Generates a single response from the loaded model. Clear context after a generation.
- `generate_chat_async` -> Generates a single response from the loaded model and keeps track of context history. 

and three signals:
- `generate_text_updated` -> Emitted during generation.
- `generate_text_finished` -> Emitted when an async generation is finished.
- `generate_text_error` -> Emitted when there is an error with text generation.

That's a quick overview, but with those three methods and those three signals you can be well on your way to using this thing. If you want to dive deeper, GDLlama includes a full suite of docs for your convenience.

- [API Reference](docs/API_REFERENCE.md): A breakdown of every function, property, and signal.
- [Architecture Guide](docs/ARCHITECTURE.md): Best practices for how to structure your code with `GDLlama`.
- [Usage Examples](docs/EXAMPLES.md) or a [Basic Example](docs/API_REFERENCE.md#example-godot-usage): Getting started, common use-cases.
- [LLM Legal](docs/AI_LEGAL.md): A curated list of legal resources for generative content in games.

# Contributions
PRs are welcome! This is my first big open source contribution and I am more than happy to share with the community. Check out [Contributing.md](docs/CONTRIBUTING.md) for more information.

Huge thanks to @Adriankhl for originally creating this project: https://github.com/Adriankhl/godot-llm
