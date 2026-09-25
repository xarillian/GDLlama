#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_generation_options.hpp"
#include "chorus/providers/llama/llama_scheduler.hpp"

namespace Chorus {

EngineCapabilities llama_provider_capabilities() {
    EngineCapabilities capabilities;
    capabilities.provider_id = "llama";
    capabilities.model_formats = {ModelFormat::Gguf};
    capabilities.input_modalities = {Modality::Text};
    capabilities.output_modalities = {Modality::Text};
    capabilities.constraint_formats = {ConstraintFormat::Gbnf, ConstraintFormat::JsonSchema};
    capabilities.scheduling = SchedulingAuthority::ChorusManaged;
    capabilities.streaming = true;
    capabilities.cancellation = true;
    capabilities.embeddings = true;
    capabilities.prompt_rendering = true;
    capabilities.message_token_counting = true;
    capabilities.common_generation_options = llama_common_generation_option_names();
    capabilities.provider_generation_options = llama_provider_generation_option_names();
    capabilities.load_options = llama_load_option_descriptors();
    return capabilities;
}

LlamaEngine::~LlamaEngine() {
    shutdown();
}

std::optional<InitializationFailure> LlamaEngine::initialize(const ChorusConfig& config, Logger logger, const InitializationControl& control) {
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

    if (control.stop_token.stop_requested())
        return InitializationFailure{ChorusError::Cancelled, "Llama initialization cancelled."};

    if (config.model.format != ModelFormat::Gguf && config.model.format != ModelFormat::Auto) {
        _log.error("Engine loads GGUF only", {{"model", config.model.model_id}});
        return InitializationFailure{ChorusError::UnsupportedModelFormat, "This engine loads GGUF models only."};
    }

    try {
        auto next_scheduler = std::make_shared<LlamaScheduler>();

        auto err = next_scheduler->initialize(config, _log, control);
        if (err.has_value()) {
            _log.error("Scheduler failed to initialize");
            return err;
        }

        if (control.stop_token.stop_requested()) {
            next_scheduler->shutdown();
            return InitializationFailure{ChorusError::Cancelled, "Llama initialization cancelled."};
        }
        {
            std::lock_guard<std::mutex> lock(_lifecycle_mutex);
            _scheduler = std::move(next_scheduler);
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        _log.error("Exception during initialization", {{"detail", e.what()}});
        return InitializationFailure{ChorusError::Unknown, e.what()};
    } catch (...) {
        _log.error("Unknown exception during initialization");
        return InitializationFailure{ChorusError::Unknown, "Llama initialization failed."};
    }
}

bool LlamaEngine::is_initialized() const {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    return _scheduler && _scheduler->is_healthy();
}

EngineCapabilities LlamaEngine::capabilities() const {
    if (auto current_scheduler = scheduler_snapshot(); current_scheduler && current_scheduler->is_healthy())
        return current_scheduler->capabilities();

    return llama_provider_capabilities();
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
    auto result = current_scheduler->render_chat_prompt(messages, template_override, enable_thinking);
    if (auto* rendered = std::get_if<RenderedPrompt>(&result))
        return std::move(*rendered);
    return std::nullopt;
}

std::optional<RequestRejection> LlamaEngine::validate_request(const ChorusRequest& request) const {
    auto current_scheduler = scheduler_snapshot();
    if (!current_scheduler || !current_scheduler->is_healthy())
        return RequestRejection{ChorusError::EngineNotReady, "LlamaEngine is not initialized."};
    return current_scheduler->validate_request(request);
}

std::shared_ptr<RequestPreparation> LlamaEngine::request_preparation() const {
    return scheduler_snapshot();
}

void LlamaEngine::submit_request(ChorusRequest chorus_request) {
    const auto id = chorus_request.id;
    const auto session = chorus_request.session_id;
    const auto on_event = chorus_request.on_event;
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(_lifecycle_mutex);
        if (_scheduler && _scheduler->is_healthy())
            accepted = _scheduler->push_request(std::move(chorus_request));
    }
    if (!accepted) {
        _log.for_request(id, session).error("Request submitted to a stopped engine");

        if (on_event) {
            ChorusSignal error_sig{
                id,
                ChorusSignal::Error{ChorusError::EngineNotReady, "Engine not initialized"},
            };
            on_event(error_sig);
        }
    }
}

void LlamaEngine::cancel_request(RequestId id) {
    std::lock_guard<std::mutex> lock(_lifecycle_mutex);
    if (_scheduler)
        _scheduler->cancel_request(id);
}

#ifdef TEST_BUILD
void LlamaEngine::set_batch_observer(std::function<void(const LlamaBatchRecord&)> observer) {
    if (auto current_scheduler = scheduler_snapshot())
        current_scheduler->set_batch_observer(std::move(observer));
}
#endif

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
