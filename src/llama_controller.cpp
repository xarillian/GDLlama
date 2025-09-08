#include "llama_controller.hpp"
#include "llama_runner.hpp"
#include <nlohmann/json.hpp>
#include <json-schema-to-grammar.h>

/**
 * @class LlamaController
 * @brief Manages the core, thread-safe logic for text generation using generic types.
 * 
 * This class is decoupled from the Godot engine and contains the primary logic for preparing and
 * executing text generation.
 * Its separation from GDLlama makes it independently testable.
 */
LlamaController::LlamaController(): 
    llama_runner(new LlamaRunner(true)), should_output_prompt(true) {}

/**
 * @brief Injects a LlamaRunner instance.
 * @param runner A std::unique_ptr to the LlamaRunner that will be used for generation.
 */
void LlamaController::set_llama_runner(std::unique_ptr<LlamaRunner> runner) {
    llama_runner = std::move(runner);
}

/**
 * @brief Sets the reverse prompt (antiprompt) for the generation.
 * @param p_reverse_prompt The string to use as the reverse prompt.
 */
void LlamaController::set_reverse_prompt(const std::string& p_reverse_prompt) {
    reverse_prompt = p_reverse_prompt;
}

/**
 * @brief Generates text synchronously and in a thread-safe manner.
 *
 * This is the core function of the controller. It acquires a lock to ensure only one generation
 * happens at a time, prepares all parameters, and then invokes the LlamaRunner.
 *
 * @param prompt The input text to generate from. Required.
 * @param grammar Optional BNF grammar string to constrain generation. Empty string for no grammar.
 * @param json Optional JSON schema to constrain generation. Will be converted to grammar 
 *             internally. If both grammar and JSON are provided, grammar takes precedence.
 * @param on_update A callback function that is invoked with each new chunk of generated text.
 * @param on_wait_start A callback function that is invoked if the model enters an interactive 
 *                      state.
 * @param on_finish A callback function that is invoked with the complete generated text when 
 *                  finished.
 * @return The complete generated text as a std::string.
 */
std::string LlamaController::generate_text_locked(
    const std::string& prompt,
    const std::string& grammar,
    const std::string& json,
    std::function<void(std::string)> on_update,
    std::function<void()> on_wait_start,
    std::function<void(std::string)> on_finish
) {
    std::lock_guard<std::mutex> lock(generate_text_mutex);

    if (!grammar.empty()) {
        params.sampling.grammar = grammar;
    } else if (!json.empty()) {
        params.sampling.grammar = json_schema_to_grammar(
            nlohmann::ordered_json::parse(json)
        );
    }

    GDLOG_DEBUG("Start Generating Text");
    llama_runner.reset(new LlamaRunner(should_output_prompt));  // @todo not this

    params.antiprompt.clear();
    if (reverse_prompt != "") {
        GDLOG_INFO("Adding reverse prompt: " + reverse_prompt);
        params.antiprompt.emplace_back(reverse_prompt);
    }

    std::string s_prompt;
    if (!params.interactive) {
        // In interactive mode, Llama automatically formats the prompt by adding the configured
        // prefix and suffix. This code ensures that non-interactive prompts are formatted
        // identically for consistent results.
        s_prompt = params.input_prefix + prompt + params.input_suffix;
    } else {
        s_prompt = prompt;
    }

    std::string generated_text = llama_runner->llama_generate_text(
        s_prompt, params, on_update, on_wait_start, on_finish
    );

    params.sampling.grammar = std::string();
    
    return generated_text;
}