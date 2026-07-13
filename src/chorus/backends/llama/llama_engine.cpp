#include "chorus/backends/llama/llama_engine.hpp"
#include "chorus/backends/llama/llama_scheduler.hpp"
#include "chorus/core/common.hpp"

#include <algorithm>
#include <iterator>
#include <variant>

namespace Chorus {

namespace {
constexpr const char* kLlamaPortableOptions[] = {"max_tokens", "temperature", "top_k", "top_p", "seed"};
constexpr const char* kLlamaBackendGenOptions[] = {"repeat_penalty"};
} // namespace

LlamaEngine::LlamaEngine() {
    // constructor can be empty -- `initialize` does the work
}

LlamaEngine::~LlamaEngine() {
    stop();
}

std::optional<ChorusError> LlamaEngine::initialize(const ChorusConfig& config) {
    _log = config.log_callback;

    if (_initialized && scheduler && scheduler->is_healthy()) {
        chorus_log(_log, LogLevel::Warn, "LlamaEngine is already initialized.");
        return std::nullopt;
    }
    if (_initialized) {
        chorus_log(_log, LogLevel::Warn, "Re-initializing LlamaEngine after engine failure.");
        stop(); // resets scheduler + _initialized; safe now that stop() is robust
    }

    if (config.model.format != ModelFormat::Gguf && config.model.format != ModelFormat::Auto) {
        chorus_log(_log, LogLevel::Error, "LlamaEngine loads GGUF only.");
        return ChorusError::UnsupportedModelFormat;
    }

    try {
        scheduler = std::make_unique<LlamaScheduler>();

        auto err = scheduler->initialize(config);
        if (err.has_value()) {
            scheduler.reset();
            chorus_log(_log, LogLevel::Error, "Failed to initialize LlamaScheduler.");
            return err;
        }

        _initialized = true;
        return std::nullopt;
    } catch (const std::exception& e) {
        chorus_log(_log, LogLevel::Error, std::string("Exception during initialization: ") + e.what());
        return ChorusError::Unknown;
    }
}

void LlamaEngine::submit_request(const ChorusRequest& chorus_request) {
    if (!_initialized || !scheduler || !scheduler->is_healthy()) {
        chorus_log(_log, LogLevel::Error, "Attempting to submit request to uninitialized engine.");

        if (chorus_request.on_event) {
            ChorusSignal error_sig;
            error_sig.request_id = chorus_request.id;
            error_sig.type = EventType::Error;
            error_sig.error_code = ChorusError::EngineNotReady;
            error_sig.text = "Engine not initialized";
            chorus_request.on_event(error_sig);
        }
        return;
    }

    scheduler->push_request(chorus_request);
}

void LlamaEngine::stop() {
    if (scheduler) {
        scheduler->stop();
        scheduler.reset();
    }
    _initialized = false;
}

bool LlamaEngine::is_initialized() const {
    return _initialized;
}

EngineCapabilities LlamaEngine::capabilities() const {
    EngineCapabilities caps;
    caps.backend_id = "llama";
    caps.model_formats = {ModelFormat::Gguf};
    caps.input_modalities = {Modality::Text};
    caps.output_modalities = {Modality::Text};
    // constraint_formats stays empty until #4 wires the grammar sampler:
    // claiming Gbnf before it constrains would violate the conformance rule.
    caps.scheduling = SchedulingAuthority::ChorusManaged;
    caps.streaming = true;
    // cancellation arrives with #4, native_sessions with #9, prompt_rendering
    // with #5, embeddings with #6.
    caps.portable_generation_options.assign(std::begin(kLlamaPortableOptions), std::end(kLlamaPortableOptions));
    caps.backend_generation_options.assign(std::begin(kLlamaBackendGenOptions), std::end(kLlamaBackendGenOptions));
    return caps;
}

std::optional<LoadedModelInfo> LlamaEngine::loaded_model_info() const {
    if (!scheduler)
        return std::nullopt;
    return scheduler->model_info();
}

std::optional<RequestRejection> LlamaEngine::validate_request(const ChorusRequest& request) const {
    if (!_initialized || !scheduler || !scheduler->is_healthy())
        return RequestRejection{ChorusError::EngineNotReady, "LlamaEngine is not initialized."};
    if (request.type == RequestType::Embedding)
        return RequestRejection{ChorusError::UnsupportedFeature, "Embeddings arrive with workstream #6."};
    const auto& c = request.gen_config.common;
    if (c.constraint)
        return RequestRejection{
            ChorusError::UnsupportedFeature, "Structured output (grammar/schema) arrives with workstream #4."
        };
    if (c.frequency_penalty || c.presence_penalty)
        return RequestRejection{
            ChorusError::UnsupportedOption, "frequency/presence penalties arrive with workstream #4."
        };
    if (!c.stop.empty())
        return RequestRejection{ChorusError::UnsupportedOption, "Stop sequences arrive with workstream #4."};
    for (const auto& [ns, value] : request.gen_config.backend_options) {
        if (ns != "llama")
            return RequestRejection{ChorusError::UnsupportedOption, "Unknown option namespace '" + ns + "'"};
        const auto* opts = std::get_if<OptionMap>(&value);
        if (!opts)
            return RequestRejection{ChorusError::UnsupportedOption, "'llama' options must be a map."};
        for (const auto& [key, v] : *opts) {
            const bool known = std::find_if(
                                   std::begin(kLlamaBackendGenOptions),
                                   std::end(kLlamaBackendGenOptions),
                                   [&](const char* k) { return key == k; }
                               ) != std::end(kLlamaBackendGenOptions);
            if (!known || !std::holds_alternative<double>(v))
                return RequestRejection{
                    ChorusError::UnsupportedOption, "Unknown or mistyped llama generation option '" + key + "'"
                };
        }
    }
    return std::nullopt;
}
} // namespace Chorus