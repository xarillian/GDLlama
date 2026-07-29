#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_scheduler.hpp"

namespace Chorus {

LlamaEngine::LlamaEngine() {
    // constructor can be empty -- `initialize` does the work
}

LlamaEngine::~LlamaEngine() {
    stop();
}

std::optional<ChorusError> LlamaEngine::initialize(const ChorusConfig& config) {
    _log = config.log_callback;

    bool already_initialized = false;
    bool holds_dead_scheduler = false;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        already_initialized = _initialized && scheduler && scheduler->is_healthy();
        // Deliberately the raw flag, not is_initialized(): a failed engine
        // reports itself uninitialized, and this is the branch that frees what
        // it still holds.
        holds_dead_scheduler = _initialized && !already_initialized;
    }
    if (already_initialized) {
        chorus_log(_log, LogLevel::Warn, "LlamaEngine is already initialized.");
        return std::nullopt;
    }
    if (holds_dead_scheduler) {
        chorus_log(_log, LogLevel::Warn, "Re-initializing LlamaEngine after engine failure.");
        stop();
    }

    if (config.model.format != ModelFormat::Gguf && config.model.format != ModelFormat::Auto) {
        chorus_log(_log, LogLevel::Error, "LlamaEngine loads GGUF only.");
        return ChorusError::UnsupportedModelFormat;
    }

    try {
        auto next_scheduler = std::make_shared<LlamaScheduler>();

        auto err = next_scheduler->initialize(config);
        if (err.has_value()) {
            chorus_log(_log, LogLevel::Error, "Failed to initialize LlamaScheduler.");
            return err;
        }

        {
            std::lock_guard<std::mutex> lock(_lifecycle_mutex);
            scheduler = std::move(next_scheduler);
            _initialized = true;
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        chorus_log(_log, LogLevel::Error, std::string("Exception during initialization: ") + e.what());
        return ChorusError::Unknown;
    }
}

void LlamaEngine::submit_request(const ChorusRequest& chorus_request) {
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        if (scheduler && scheduler->is_healthy())
            accepted = scheduler->push_request(chorus_request);
    }
    if (!accepted) {
        chorus_log(_log, LogLevel::Error, "Attempting to submit request to uninitialized engine.");

        if (chorus_request.on_event) {
            ChorusSignal error_sig;
            error_sig.request_id = chorus_request.id;
            error_sig.type = EventType::Error;
            error_sig.error_code = ChorusError::EngineNotReady;
            error_sig.text = "Engine not initialized";
            chorus_request.on_event(error_sig);
        }
    }
}

void LlamaEngine::cancel_request(RequestId id) {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    if (scheduler)
        scheduler->cancel_request(id);
}

void LlamaEngine::stop() {
    std::shared_ptr<LlamaScheduler> stopped_scheduler;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        stopped_scheduler = std::move(scheduler);
        _initialized = false;
    }
    if (stopped_scheduler)
        stopped_scheduler->stop();
}

bool LlamaEngine::is_initialized() const {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    // A worker that hit a fatal decode stops its scheduler without touching
    // _initialized, so the flag alone would keep claiming readiness for an
    // engine that rejects everything sent to it.
    return _initialized && scheduler && scheduler->is_healthy();
}

EngineCapabilities LlamaEngine::capabilities() const {
    EngineCapabilities caps;
    caps.provider_id = "llama";
    caps.model_formats = {ModelFormat::Gguf};
    caps.input_modalities = {Modality::Text};
    caps.output_modalities = {Modality::Text};
    caps.constraint_formats = {ConstraintFormat::Gbnf, ConstraintFormat::JsonSchema};
    caps.scheduling = SchedulingAuthority::ChorusManaged;
    caps.streaming = true;
    caps.cancellation = true;
    caps.prompt_rendering = true; // #5
    // native_sessions arrives with #9, embeddings with #6.
    caps.common_generation_options = llama_common_generation_option_names();
    caps.provider_generation_options = llama_provider_generation_option_names();
    caps.load_options = llama_load_option_descriptors();
    return caps;
}

std::optional<LoadedModelInfo> LlamaEngine::loaded_model_info() const {
    std::shared_ptr<LlamaScheduler> current_scheduler;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        current_scheduler = scheduler;
    }
    if (!current_scheduler)
        return std::nullopt;
    return current_scheduler->model_info();
}

std::optional<RequestRejection> LlamaEngine::validate_request(const ChorusRequest& request) const {
    std::shared_ptr<LlamaScheduler> current_scheduler;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        current_scheduler = scheduler;
    }
    if (!current_scheduler || !current_scheduler->is_healthy())
        return RequestRejection{ChorusError::EngineNotReady, "LlamaEngine is not initialized."};
    if (request.type == RequestType::Embedding)
        return RequestRejection{ChorusError::UnsupportedFeature, "Embeddings arrive with workstream #6."};
    return validate_llama_request(request);
}

std::optional<RenderedPrompt> LlamaEngine::render_chat_prompt(
    const std::vector<ChatMessage>& messages, const std::string& template_override, bool enable_thinking
) const {
    std::shared_ptr<LlamaScheduler> current_scheduler;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        current_scheduler = scheduler;
    }
    if (!current_scheduler || !current_scheduler->is_healthy())
        return std::nullopt;
    return current_scheduler->render_chat_prompt(messages, template_override, enable_thinking);
}
} // namespace Chorus
