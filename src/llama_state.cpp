#include "llama_state.hpp"
#include "logging_utils.hpp"

LlamaState::LlamaState() : model(nullptr), model_context(nullptr), is_model_loaded(false) {}

LlamaState::~LlamaState() {
    unload();
}

bool LlamaState::load(common_params & params) {
    if (is_model_loaded) {
        GDLOG_WARN("A model is already loaded. GDLlama will unload it before loading the new one.");
        unload();
    }

    GDLOG_DEBUG("Loading model from path: " + params.model.path);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_init_result = common_init_from_params(params);
    model = llama_init_result.model.get();
    model_context = llama_init_result.context.get();

    if (model == nullptr) {
        GDLOG_ERROR("Failed to load model.");
        llama_backend_free();
        return false;
    }

    if (model_context == nullptr) {
        GDLOG_ERROR("Failed to create context.");
        llama_backend_free();
        return false;
    }

    is_model_loaded = true;
    GDLOG_DEBUG("Model loaded successfully.");
    return true;
}

void LlamaState::unload() {
    if (!is_model_loaded) {
        GDLOG_DEBUG("Model is not loaded.");
        return;
    }

    GDLOG_DEBUG("Unloading model...");

    llama_init_result = {};

    llama_backend_free();

    model = nullptr;
    model_context = nullptr;
    is_model_loaded = false;

    GDLOG_DEBUG("Model unloaded.");
}

bool LlamaState::is_loaded() const {
    return is_model_loaded;
}

llama_context* LlamaState::get_context() {
    return model_context;
}

llama_model* LlamaState::get_model() {
    return model;
}