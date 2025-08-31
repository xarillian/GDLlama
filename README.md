# Godot LLM
> Isn't it cool to utilize large language model (LLM) to generate contents for your game?
- @Adriankhl, original creator of Godot LLM

Why, yes, I think it is cool! LLMs and multimodal models do have great potential for multiple aspects of game design. Thanks to `llama.cpp`, we have local inference fast enough to enable some genuinely interesting gameplay and I want Godot at the forefront of that. Or at least, not lagging behind the big guns.

This is a fork of [Adriankhl's original godot-llm](https://github.com/Adriankhl/godot-llm) with updated build instructions and fixes for recent llama.cpp versions.

# Getting Started
Sorry! You gotta build everything yourself right now. This is not in the asset library, no sir. In fact, I'm only 70-80% sure it works at all! That means: no real quick starts. Not yet! I'll try to upload this to the Asset Library.

I'm a Windows scrub, so it is only tested there. I'll have to pull out my Linux machine at some point to run through this.

## Prerequisites
- CMake 3.14+
- Ninja build system
- Vulkan SDK (for GPU builds)
- Git
- (for Windows): Visual Studio Build Tools with clang-cl
    - or some equivalent

# Building
## Initial
Install the necessary build tools (e.g. `clang`) and Vulkan SDK for your operating system, then clone this repository.

```shell
git clone https://github.com/xarillian/GDLlama.git
cd godot-llm
git submodule update --init --recursive
mkdir build
cd build
```

## Run `cmake`
### Windows
from preset (recommended):
```shell
cmake --preset windows-vulkan-release ..
```

or manually:
```shell
cmake .. -GNinja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_CXX_FLAGS="/EHsc" -DLLAMA_NATIVE=OFF -DLLAMA_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_COMMON=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release
```

### Linux
I haven't tested this at all, sorry. Here's the advice from the original project:

```shell
cmake .. -GNinja -DLLAMA_NATIVE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DLLAMA_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
```

### Android
I haven't tested this at all, sorry. Here's the advice from the original project:

For Android, set `$NDK_PATH` to your android ndk directory, then:

```shell
cmake .. -GNinja -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH\cmake\android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS="-mcpu=generic" -DCMAKE_CXX_FLAGS="-mcpu=generic" -DCMAKE_BUILD_TYPE=Release
```

## Ninja
Then compile and install by `ninja`:

```shell
ninja -j4
ninja install
```

## Final Steps
The folder `godot-llm/install/gpu/addons/godot_llm` can be copied to the `addons` folder of your Godot project. On Windows at least, you will also need t ocopy the required DLL dependencies:
- `ggml.dll`
- `ggml-base.dll`
- `ggml-cpu.dll`
- `llama.dll`

These are located in `godot-llm/install/bin`. Copy them into your Godot project's `addons/godot_llm/bin/` directory.

Replace "gpu" with "cpu" for a CPU build.

# Contributions
- PRs are welcome! This is my first big open source contribution and I am more than happy to share with the community.
- Huge thanks to @Adriankhl for originally creating this project. See: https://github.com/Adriankhl/godot-llm

# Documentation

**Note:** All of this is suspect right now. I haven't reviewed _any_ of it!

## Inspector Properties: GDLlama, GDEmbedding, and GDLlava
There are 3 base nodes added by this plugin: `GdLlama`, `GdEmbedding`, and `GdLlava`.
Each type of node owns a set of properties which affect the computational performance and the generated output. Some of the properties belong to more than one node, and they generally have similar meaning for all types of node.

* `Model Path`: location of your GGUF model
* `Mmproj Path` location of your `mmproj` GGUF file, for `GdLlava` only
* `Instruct`: question and answer interactive mode
* `Interactive`: custom interactive mode, you should set your `reverse_prompt`, `input_prefix`, and `input_suffix` to set up a smooth interaction
* `Reverse Prompt`: AI stops to wait for user input after seeing this prompt being generated, a good example is "User:"
* `Input Prefix`: append before every user input
* `Input Suffix`: append after every user input
* `Should Output prompt`: whether the input prompt should be included in the output
* `Should Output Special`: whether the special (e.g., beginning of sequence and ending of sequence) token should be included in the output
* `Context Size`: number of tokens the model can process at a time
* `N Predict`: number of new tokens to generate, generate infinite sequence if -1
* `N Keep`: when the model run out of `context size`, it starts to forget about earlier context, set this variable to force the model to keep a number of the earliest tokens to keep the conversation relevant
* `Temperature`: the higher the temperature, the more random the generated text
* `Penalty Repeat`: penalize repeated sequence, diabled if -1
* `Penalty Last N`: the number of latest token to consider when penalizing repeated sequence, disabled if 0, `Context Size` if -1
* `Penalilze Nl`: penallize newline token
* `Top K`: only sample from this amount of tokens with the highest probabilities, disabled if 0
* `Top P`: only sample from tokens within this cumulative probability, disabledd if 1.0
* `Min P`: only sample from tokens with at least this probability, disabledd if 0.0
* `N Thread`: number of cpu threads to use
* `N GPU Layer`: number of layer offloaded to GPU
* `Main GPU`: the main GPU for computation
* `Split Mode`: how the computation will be distributed if there are multiple GPU in your systemm (0: None, 1: Layer, 2: Row)
* `Escape`: process escape character in input prompt
* `N Batch`: maximum number of tokens per iteration during continuous batching
* `N Ubatch`: maximum batch size for computation

## GdLlama functions and signals

### Functions
* `generate_text_simple(prompt: String) -> String`: generate text from prompt
* `generate_text_json(prompt: String, json: String) -> String`: generate text in a format enforced by a [json schema](https://json-schema.org/), see [the following section](#text-generation-with-json-schema)
* `generate_text_grammar(prompt: String, grammar: String) -> String`: generate text in a format enforced by [GBNF grammar](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)
* `generate_text(prompt: String, grammar: String, json: String) -> String`: a wrapper function, run `generate_text_gramma` if `grammar` is non-empty, runs `generate_text_json` if `json` is non-empty, run `generate_text_simple` otherwise
* `run_generate_text(prompt: String, grammar: String, json: String) -> Error`: run `generate_text` in background, rely on signals to recieve generated text, note that only one background thread is allowd for a GDLlama node, calling this function when the background thread is still running will freeze the logic until the background thread is done
* `input_text(input: String)`: input text to interactively generate text (with either `Instruct` or `Interactive` enabled) with the model, only works if the model is waiting for intput, inputing an empty string means the model should continue to generate what it has been generating
* `stop_generate_text()`: stop text generation, clean up the model and the background thread
* `is_running() -> bool`: whether the background thread is running
* `is_waiting_input() -> bool`: whether the model is waiting for input text (with either `Instruct` or `Interactive` enabled)

### Signals
* `generate_text_finished(text: String) `: emitted with the full generated text when a text generation is completed. When either `Instruct` or `Interactive` enabled, this signal is emitted after the whole interaction is finished
* `generate_text_updated(new_text: String)`: instead of waiting the full generated text, this signal is emited whenever a new token (part of the text sequence) is generated, which forms a stream of strings
* `input_wait_started()`: the model is now starting to wait for user input, happens when either `Instruct` or `Interactive` are enabled and the model stop generating text in the middle of the conversation to wait for further input from the user.

## GDEmbedding functions and signals
### Functions
* `compute_embedding(prompt: String) -> PackedFloat32Array`: compute the embedding vector of a prompt
* `similarity_cos_array(array1: PackedFloat32Array, array2: PackedFloat32Array) -> float`: compute the cosine similarity between two embedding vectors, this is a fast function, no model is loaded
* `similarity_cos_string(s1: String, s2: String) -> float`: compute the cosine similarity between two strings
* `run_compute_embedding(prompt: String) -> Error`: run `compute_embedding(prompt: String)` in background, rely on the `compute_embedding_finished` signal to recieve the embedding vector, note that only one background thread is allowd for a GDEmbedding node, calling this function when the background thread is still running will freeze the logic until the background thread is done
* `run_similarity_cos_string(s1: String, s2: String) -> Error`: run `similarity_cos_string` in background, rely on the `compute_similairty_finished` signal to recieve the cosine similairty, note that only one background thread is allowd for a GDEmbedding node, calling this function when the background thread is still running will freeze the logic until the background thread is done
* `is_running() -> bool`: whether the background thread is running

### Signals
* `compute_embedding_finished(embedding: PackedFloat32Array)`: emitted when `run_compute_embedding` is completed
* `similarity_cos_string_finished(similarity: float)`: emitted when `run_similarity_cos_string` is completed

## GDLlava functions and signals
### functions
* `generate_text_base64(prompt: String, image_base64: String) -> String`: generate text based on a prompt and a base64 string which encodes a `jpg` or `png` image
* `generate_text_image(prompt: String, image: Image) -> String`: generate text based on a prompt and an `Image` object in Godot
* `run_generate_text_base64(prompt: String, image_base64: String) -> Error`: run `generate_text_base64` in background, rely on signals to recieve generated text, note that only one background thread is allowd for a GDLlava node, calling this function when the background thread is still running will freeze the logic until the background thread is done
* `run_generate_text_base64(prompt: String, image: Image) -> Error`: run `generate_text_base64` in background, rely on signals to recieve generated text, note that only one background thread is allowd for a GDLlava node, calling this function when the background thread is still running will freeze the logic until the background thread is done
* `stop_generate_text()`: stop text generation, clean up the model and the background thread
* `is_running() -> bool`: whether the background thread is running

### Signals
* `generate_text_finished(text: String) `: emitted with the full generated text when a text generation is completed
* `generate_text_updated(new_text: String)`: instead of waiting the full generated text, this signal is emited whenever a new token (part of the text sequence) is generated, which forms a stream of strings


## Text generation with Json schema

Suppose you want to generate a character with:
  * `name`: a string from 3 character to 20 character
  * `birthday`: a string with a specific date format
  * `weapon`: either "sword", "bow", or "wand
  * `description`: a text with minimum 10 character

You should first create a GDLlama node, and turn `Should Output prompt` and `Should Output Special` off either by inspector or by script:
```
should_output_prompt = false
should_output_special = false
```


Construct the following `_person_schema` dictionary in GDScript:
```
var _person_schema = {
	"type": "object",
	"properties": {
		"name": {
			"type": "string",
			"minLength": 3,
			"maxLength": 20,
		},
		"birthday": {
			"type": "string",
			"format": "date"
		},
		"weapon": {
			 "enum": ["sword", "bow", "wand"],
		},
		"description": {
			"type": "string",
			"minLength": 10,
		},
	},
	"required": ["name", "birthday", "weapon", "description"]
}
```

Then convert it to a json string
```
var person_schema: String = JSON.stringify(_person_schema)
```

Supposed you are interested in a "Main character in a magic world", you can generate the character using the `generate_text_json(prompt, json_scheme)` of the `GDLlama` node:
```
var json_string: String = generate_text_json(prompt, json_scheme)
```

Note that text generation is slow, you may want to use `run_generate_text(prompt, "", json_scheme)` to run the generation in background, then handle `generate_text_finished` to receive the generated text.

`json_string` should look like this:
```
{"birthday": "2000-05-12", "description": "A young wizard with a pure heart and a mischievous grin. He has a wild imagination and a love for adventure. He is always up for a challenge and is not afraid to take risks.", "name": "Eryndor Thorne", "weapon": "wand"}
```

Now, the generated data is ready, you can parse back to a dictionary or other object to use the data.
```
var dict: Dictionary = {}
var json = JSON.new()
var error = json.parse(json_string)
if (error == OK):
		dict = json.data

print(dict["name"]) ##Eryndor Thorne
```

## Inspector properties: LlmDB
LlmDB extends GDEmbedding and shares all its properties, check the section above for the relevant information. Additionally, LlmDB has
* `Meta`: an array of LlmDBMetaData Resource which defines the structure of the metadata. LlmDBMetaData contains `Data Name` which define the name of a metadata, and `Data Type` (0=integer, 1=real, 2=text, 3=blob) to define the data type of the metadata. `Meta` should be non-empty, and the first element of `Meta` should always be an `id` with text as the `Data Type`.
* `dB Dir`: the directory of the database file, default is the root directory of the project
* `dB File`: the file name of the database file, default is `llm.db`
* `Table Name`: defines the name of the tables created by the `create_llm_tables` function
* `Embedding Size`: the vector size of the embedding computed by the model, used in the `create_llm_tables` function
* `Absolute Separators`: an array of `String`. When storing a piece of text, the text will be first separated by the `String` defines here, the separation process will stop if the separated text is shorter than `Chunk Size` or all the separators here have been processed. The default are `\n` and `\n\n`, which are displayed as empty space in the inspector.
* `Chunk Separators`: an array of `String`. After the `Absolute Separators` are processed, one of the separators (first one that works) here will be chosen to further separated the piece of texts, then the pieces are grouped up to chunks to fulfill the requirements of `Chunk Size` and `Chunk Overlap`
* `Chunk Size`: any text chunk should not exceed this size, unless the separation function fails to fulfill the requirement after iteratoring through the iterators
* `Chunk Overlap`: the maximum overlap between neighbouring text chunks, the algorithm will try to create the biggest overlap possible fulfilling this constraint

## LlmDB Functions and Signals

Besides the functions and signals from GDEmbedding, LlmDB has a few more functions and signals

### Functions

* `calibrate_embedding_size()`: calibrate `Embedding Size` to the correct number based on the model in `model_path`
* `open_db()`: create a `dB_File` at `dB_Dir` if the file doesn't exist, then connect to the database
* `close_db()`: terminate the connection to the database
* `execute(statement: String)` execute an sql statement, turn on `Verbose stdout` in `Project Settings` to see the log generated by this statement
* `create_llm_tables()`: create a table with name `Table Name` if the table doesn't exist, a `Table Name` + `_meta` table to store pre-defined metadata by `id`, and some `_virtual` tables to 
* `drop_table(p_table_name: String)`: drop a table with a specific name
* `drop_llm_tables(p_table_name: String)`: drop all tables (except the `sqlite_sequence` table which is created automatically for autoincrement) created by `create_llm_tables()`, i.e., `p_table_name`, `p_table_name` + `_meta` and every table with a name containing `p_table_name` + `_virtual`
* `has_table(p_table_name: String) -> bool`: whether a table with this name exists
* `is_table_valid(p_table_name: String) -> bool`: whether the table contains valid metadata, i.e., all elements in `.meta` properties exist in the table and the data types are correct
* `store_meta(meta_dict: Dictionary)`: store a set of meta data to table `Table Name` + `_meta` with `id` as the primary key, such that you can call `store_text_by_id` by id instead of inputting the full metadata dictionary through `store_text_by_meta`
* `has_id(id: String, p_table_name: String) -> bool`: whether the table has a specific id stored
* `split_text(text: String) -> PackedStringArray`: split a piece of text first by all `Absolute Separators`, then by one of the appropiate `Chunk Separators`, such that any text chunk is shorter than `Chunk Size` (measured in character), and the overlap is close to but not greater than `Chunk Overlap`. If the algorithm failed to satisfy the contraints, there will be an error message printed out and the returned chunk will be greater than the `Chunk Size`
* `store_text_by_id(id: String, text: String)`: split the text and store the chunks in the database, be aware that `store_meta` should have been called previously such that the `id` with the corresponding meta is already in the database
* `run_store_text_by_id(id: String, text: String) -> Error`: run `store_text_by_id` in background, emits `store_text_finished` signal when finished
* `store_text_by_meta(meta_dict: Dictionary, text: String)`: split the text and store the chunks in the database with the metadata defined in `meta_dict`, be aware that the metadata should be valid, every key should be a name stored in the `.meta` property and the corresponding type should be correct 
* `run_store_text_by_meta(meta_dict: Dictionary, text: String) -> Error` run `store_text_by_meta` in background, emits `store_text_finished` signal when finished
* `retrieve_similar_texts(text: String, where: String, n_results: int) -> PackedStringArray`: retrieve `n_results` most similar text chunks to `text`, `where` should be empty or an sql WHERE clause to filter the chunks by metadata
* `run_retrieve_similar_texts(text: String, where: String, n_results: int) -> Error`:
run `retrieve_similar_texts` in background, and emits a `retrieve_similar_texts_finished` signal once it is done

### Signals

* `store_text_finished`: emitted when `run_store_text_by_id` or `run_store_text_by_meta` is finished
* `retrieve_similar_texts_finished(array: PackedStringArray)`: contains an array of `String`, emitted when `run_retrieve_similar_texts` is finished

## LlmDBMetaData

This is a simple resource class that forms the `meta` array property in LlmDB. It has two properties:

* `data_name`: a `String` that defines the name of this metadata
* `data_type`: an `int` that defines the data type of this metadata (0=integer, 1=real, 2=text, 3=blob), note that inputing an integer here is not recommended since it can be confusing, use the inspector properties, the LlmDBMetaData enum or the function below instead
* `LlmDBMetaDataType` enum:
    - `LlmDBMetaData.INTEGER = 0`
    - `LlmDBMetaData.REAL = 1`
    - `LlmDBMetaData.TEXT = 2`
    - `LlmDBMetaData.BLOB = 3`

There are 4 static functions to create LlmDBMetaData

* `create_int(data_name: String) -> LlmDBMetaData`: create a LlmDBMetaData with type int (0)
* `create_real(data_name: String) -> LlmDBMetaData`: create a LlmDBMetaData with type real (1)
* `create_text(data_name: String) -> LlmDBMetaData`: create a LlmDBMetaData with type text (2)
* `create_blob(data_name: String) -> LlmDBMetaData`: create a LlmDBMetaData with type blob (3), note that blob data type support is still a work-in-progress

Alternatively, you can use this static function to create LlmDBMetaData
* `create(data_name: String, data_type: int) -> LlmDBMetaData`: create a corresponding LlmDBMetaData by `data_name` and `data_type`, it is recommended to use the enum instead of `int` for `data_type`

# FAQ

1. How to get more debug message?

Turn on `Verbose stdout` in `Project Settings`, consider running Godot from a terminal to get additional logging messages.

2. Does it support languages other than English?

Yes, the plugin uses utf8 encoding so it has multilingual support naturally. However, a language model may be trained with English data only and it won't be able to generate text other than English, choose the language model based on your need.

3. Strange tokens in generated text, such as `<eot_id>` when `Should Output Special` is off.

You are always welcome to open an issue. However, be aware that the standard of GGUF format can be changed to support new features and models, such that the bug can come from the model side instead of within this plugin. For example, some older llama 3 GGUF model may not be compatible with the latest format, you may try to search for a newer model with fixes such as [this](https://huggingface.co/NikolayKozloff/Meta-Llama-3-8B-Instruct-bf16-correct-pre-tokenizer-and-EOS-token-Q8_0-Q6_k-Q4_K_M-GGUF/tree/main).

4. You are running Arch linux (or its derivatives such as Manjaro) and your Godot Editor crash.

The Arch build of Godot is bugged when working with GDExtension, download Godot from the official website instead.

5. You have a discrete GPU and you see `unable to load model` error, you have make sure that the model parameters are correctly set.

There is currently a bug on vulkan backend if you have multiple drivers installed for the same GPU, try to turn `Split Mode` to `NONE` (0) and set your `Main GPU` manually (starting from 0) to see if it works.

