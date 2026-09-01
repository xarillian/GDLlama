#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_generation_options.hpp"
#include "chorus/providers/llama/llama_scheduler.hpp"

namespace Chorus {

LlamaEngine::~LlamaEngine() {
    shutdown();
}

std::optional<ChorusError> LlamaEngine::initialize(const ChorusConfig& config, Logger logger) {
    bool holds_dead_scheduler = false;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        if (_scheduler && _scheduler->is_healthy())
            return std::nullopt;
        holds_dead_scheduler = static_cast<bool>(_scheduler);
    }

    _log = std::move(logger);
    if (holds_dead_scheduler) {
        _log.warn("Re-initializing engine after a failure");
        shutdown();
    }

    if (config.model.format != ModelFormat::Gguf && config.model.format != ModelFormat::Auto) {
        _log.error("Engine loads GGUF only", {{"model", config.model.model_id}});
        return ChorusError::UnsupportedModelFormat;
    }

    try {
        auto next_scheduler = std::make_shared<LlamaScheduler>();

        auto err = next_scheduler->initialize(config, _log);
        if (err.has_value()) {
            _log.error("Scheduler failed to initialize");
            return err;
        }

        {
            std::lock_guard<std::mutex> lock(_lifecycle_mutex);
            _scheduler = std::move(next_scheduler);
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        _log.error("Exception during initialization", {{"detail", e.what()}});
        return ChorusError::Unknown;
    }
}

bool LlamaEngine::is_initialized() const {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    return _scheduler && _scheduler->is_healthy();
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
    caps.prompt_rendering = true;
    caps.common_generation_options = llama_common_generation_option_names();
    caps.provider_generation_options = llama_provider_generation_option_names();
    caps.load_options = llama_load_option_descriptors();
    return caps;
}

std::optional<LoadedModelInfo> LlamaEngine::loaded_model_info() const {
    auto current_scheduler = scheduler_snapshot();
    if (!current_scheduler)
        return std::nullopt;
    return current_scheduler->model_info();
}

std::optional<RenderedPrompt> LlamaEngine::render_chat_prompt(
    const std::vector<ChatMessage>& messages, const std::string& template_override, bool enable_thinking
) const {
    auto current_scheduler = scheduler_snapshot();
    if (!current_scheduler || !current_scheduler->is_healthy())
        return std::nullopt;
    return current_scheduler->render_chat_prompt(messages, template_override, enable_thinking);
}

std::optional<RequestRejection> LlamaEngine::validate_request(const ChorusRequest& request) const {
    auto current_scheduler = scheduler_snapshot();
    if (!current_scheduler || !current_scheduler->is_healthy())
        return RequestRejection{ChorusError::EngineNotReady, "LlamaEngine is not initialized."};
    if (request.type == RequestType::Embedding)
        return RequestRejection{ChorusError::UnsupportedFeature, "LlamaEngine does not produce embeddings."};
    return validate_llama_request(request);
}

void LlamaEngine::submit_request(const ChorusRequest& chorus_request) {
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        if (_scheduler && _scheduler->is_healthy())
            accepted = _scheduler->push_request(chorus_request);
    }
    if (!accepted) {
        _log.for_request(chorus_request.id, chorus_request.session_id).error("Request submitted to a stopped engine");

        if (chorus_request.on_event) {
            ChorusSignal error_sig{
                chorus_request.id,
                ChorusSignal::Error{ChorusError::EngineNotReady, "Engine not initialized"},
            };
            chorus_request.on_event(error_sig);
        }
    }
}

void LlamaEngine::cancel_request(RequestId id) {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    if (_scheduler)
        _scheduler->cancel_request(id);
}

void LlamaEngine::shutdown() {
    std::shared_ptr<LlamaScheduler> stopped_scheduler;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        stopped_scheduler = std::move(_scheduler);
    }
    // Stop publishing the scheduler before waiting for its worker and callbacks,
    // without holding the lifecycle mutex across that wait.
    if (stopped_scheduler)
        stopped_scheduler->shutdown();
}

std::shared_ptr<LlamaScheduler> LlamaEngine::scheduler_snapshot() const {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    return _scheduler;
}
} // namespace Chorus
