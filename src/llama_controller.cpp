#include "llama_controller.hpp"
#include <nlohmann/json.hpp>
#include <common/json-schema-to-grammar.h>
#include <winerror.h>

LlamaController::LlamaController() : 
    llama_state(std::make_unique<LlamaState>()),
    llama_runner(std::make_unique<LlamaRunner>()) {}

std::string LlamaController::start_generation(
    common_params& params,
    const std::string& prompt,
    const std::string& grammar,
    const std::string& json,
    bool is_continuous,
    std::function<void(std::string)> on_update
) {

    if (!is_model_loaded()) {
        std::string err_msg = "Cannot generate text: Model is not loaded.";
        GDLOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
    }

    if (!is_continuous) {
        reset_context();
    }

    params.prompt = prompt;
    if (!grammar.empty()) {
        params.sampling.grammar = grammar;
    } else if (!json.empty()) {
        params.sampling.grammar = json_schema_to_grammar(nlohmann::ordered_json::parse(json));
    }

    llama_context* ctx = llama_state->get_context();
    llama_model* model = llama_state->get_model();

    std::string generated_text = llama_runner->run_prediction(model, ctx, params, on_update);

    params.sampling.grammar.clear();

    return generated_text;
}

void LlamaController::stop_generation() {
    if (llama_runner) {
        llama_runner->stop_generation();
    }
}

bool LlamaController::is_model_loaded() const {
    return llama_state->is_loaded();
}

void LlamaController::reset_context() {
    if (is_model_loaded()) {
        llama_memory_clear(llama_get_memory(llama_state->get_context()), true);
        GDLOG_DEBUG("LLM context (KV cache) cleared.");
    }
}

godot::Error LlamaController::load_model(common_params& params) {
    if (params.model.path.empty()) {
        GDLOG_ERROR("Cannot load model: model_path is not set.");
        return godot::FAILED;
    }
    bool success = llama_state->load(params);
    godot::Error status = success ? godot::OK : godot::FAILED;
    GDLOG_DEBUG("Load model status: " + status);
    return status;
}

void LlamaController::unload_model() {
    llama_state->unload();
}
