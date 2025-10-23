#include "llama_controller.hpp"
#include <nlohmann/json.hpp>
#include <common/json-schema-to-grammar.h>

LlamaController::LlamaController() : 
    llama_state(std::make_unique<LlamaState>()),
    llama_runner(std::make_unique<LlamaRunner>()) {}

std::string LlamaController::start_generation(
    common_params& params,
    const std::string& prompt,
    const std::string& grammar,
    const std::string& json,
    bool is_conversational,
    std::function<void(std::string)> on_update,
    std::string* error_msg
) {

    if (!is_model_loaded()) {
        std::string err = "Cannot generate text: Model is not loaded.";
        GDLOG_ERROR(err);
        if (error_msg) *error_msg = err;
        return "";
    }

    if (is_conversational) {
        conversation_history.push_back({"user", prompt});
    } else {
        reset_context();
        params.prompt = prompt;
    }

    if (!grammar.empty()) {
        params.sampling.grammar = grammar;
    } else if (!json.empty()) {
        params.sampling.grammar = json_schema_to_grammar(nlohmann::ordered_json::parse(json));
    }

    llama_context* ctx = llama_state->get_context();
    llama_model* model = llama_state->get_model();

    std::string generated_text = llama_runner->run_prediction(
        model, 
        ctx, 
        params,
        is_conversational ? &conversation_history : nullptr,
        on_update,
        error_msg
    );

    if (error_msg && !error_msg->empty()) {
        return "";  // Error occurred in runner
    }

    if (is_conversational) {
        // We specifically add empty assistant messages to the history here.
        // These could be ignored, but it's useful to see where the model responded
        // in the conversation.
        conversation_history.push_back({ "assistant", generated_text.c_str() });
    }

    params.sampling.grammar.clear();

    return generated_text;
}


std::vector<float> LlamaController::generate_embedding(
    common_params& params,
    const std::string& prompt,
    std::string* error_msg
) {
    if (!is_model_loaded()) {
        std::string err = "Cannot generate embedding: Model is not loaded.";
        GDLOG_ERROR(err);
        if (error_msg) *error_msg = err;
        return {};
    }

    params.n_predict = 0;
    params.prompt = prompt;

    llama_context* ctx = llama_state->get_context();
    llama_model* model = llama_state->get_model();

    reset_context();

    std::vector<float> embedding = llama_runner->run_embedding(model, ctx, params, error_msg);

    if (error_msg && !error_msg->empty()) {
        return {};  // Error occurred in runner
    }

    return embedding;
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
        llama_memory_t mem = llama_get_memory(llama_state->get_context());
        llama_memory_clear(mem, true);
        GDLOG_DEBUG("LLM context sequence reset.");
    }
    conversation_history.clear();
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
    reset_context();
    GDLOG_DEBUG("Chat history cleared on model unload.");
    llama_state->unload();
    GDLOG_DEBUG("Model unloaded.");
}

std::vector<ChatMessage> LlamaController::get_conversation_history() const {
    return conversation_history;
}