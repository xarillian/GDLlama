#include "llama_controller.hpp"
#include <nlohmann/json.hpp>
#include <json-schema-to-grammar.h>

LlamaController::LlamaController() : llama_runner(new LlamaRunner()) {}

std::string LlamaController::start_generation(
    llama_model* model,
    llama_context* context,
    common_params& params,
    const std::string& prompt,
    const std::string& grammar,
    const std::string& json,
    std::function<void(std::string)> on_update,
    std::function<void(std::string)> on_finish
) {
    params.prompt = prompt;

    if (!grammar.empty()) {
        params.sampling.grammar = grammar;
    } else if (!json.empty()) {
        params.sampling.grammar = json_schema_to_grammar(
            nlohmann::ordered_json::parse(json)
        );
    }

    std::string generated_text = llama_runner->run_prediction(
        model,
        context,
        params,
        on_update,
        on_finish
    );

    params.sampling.grammar.clear();

    return generated_text;
}

void LlamaController::stop_generation() {
    if (llama_runner) {
        llama_runner->stop_generation();
    }
}
