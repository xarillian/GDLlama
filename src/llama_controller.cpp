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

    if (is_continuous) {
        conversation_history.push_back({"user", prompt});

        std::vector<llama_chat_message> messages_for_api;
        for (const auto& msg : conversation_history) {
            messages_for_api.push_back({msg.role.c_str(), msg.content.c_str()});
        }

        const int32_t buffer_size = llama_n_ctx(llama_state->get_context());
        std::vector<char> buffer(buffer_size);

        int32_t formatted_size = llama_chat_apply_template(
            params.chat_template.empty() ? nullptr : params.chat_template.c_str(),
            messages_for_api.data(),
            messages_for_api.size(),
            true, // add_assistant_prefix
            buffer.data(),
            buffer.size()
        );
        
        if (formatted_size < 0) {
            conversation_history.pop_back();
            throw std::runtime_error("Failed to apply chat template.");
        }

        if (static_cast<int32_t>(buffer.size()) <= formatted_size) {
            conversation_history.pop_back();
            throw std::runtime_error("Formatted chat prompt exceeds the buffer size.");
        }

        params.prompt = std::string(buffer.data());
    } else {
        // For non-continuous (single-shot) generation, we don't apply a chat
        // template. This mode is for raw text completion, and applying chat
        // formatting would be incorrect.
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

    std::string generated_text = llama_runner->run_prediction(model, ctx, params, on_update);

    if (is_continuous && !generated_text.empty()) {
        conversation_history.push_back({ "assistant", generated_text.c_str() });
    }

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
