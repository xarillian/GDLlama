#include "chorus/backends/llama/llama_engine.hpp"
#include "chorus/backends/llama/llama_generation.hpp"
#include "chorus/backends/llama/llama_scheduler.hpp"
#include "chorus/core/common.hpp"

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
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        already_initialized = _initialized && scheduler && scheduler->is_healthy();
    }
    if (already_initialized) {
        chorus_log(_log, LogLevel::Warn, "LlamaEngine is already initialized.");
        return std::nullopt;
    }
    if (is_initialized()) {
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
    return _initialized;
}

EngineCapabilities LlamaEngine::capabilities() const {
    EngineCapabilities caps;
    caps.backend_id = "llama";
    caps.model_formats = {ModelFormat::Gguf};
    caps.input_modalities = {Modality::Text};
    caps.output_modalities = {Modality::Text};
    caps.constraint_formats = {ConstraintFormat::Gbnf, ConstraintFormat::JsonSchema};
    caps.scheduling = SchedulingAuthority::ChorusManaged;
    caps.streaming = true;
    caps.cancellation = true;
    // native_sessions arrives with #9, prompt_rendering with #5, embeddings with #6.
    caps.portable_generation_options = llama_portable_generation_option_names();
    caps.backend_generation_options = llama_backend_generation_option_names();
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
    return validate_llama_generation(request.gen_config);
}
} // namespace Chorus
