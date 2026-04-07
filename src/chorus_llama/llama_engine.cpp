#include "chorus_llama/llama_engine.hpp"
#include "chorus_core/chorus_common.hpp"
#include "chorus_llama/llama_scheduler.hpp"

namespace Chorus {
LlamaEngine::LlamaEngine() {
    // constructor can be empty -- `initialize` does the work
}

LlamaEngine::~LlamaEngine() {
    stop();
}

bool LlamaEngine::initialize(const ChorusConfig& config) {
    _log = config.log_callback;

    if (_initialized) {
        chorus_log(_log, LogLevel::Warn, "LlamaEngine is already initialized.");
        return true;
    }

    if (config.model_path.empty()) {
        chorus_log(_log, LogLevel::Error, "Model path is empty.");
        return false;
    }

    try {
        scheduler = std::make_unique<LlamaScheduler>();

        if (!scheduler->initialize(config)) {
            scheduler.reset();
            chorus_log(_log, LogLevel::Error, "Failed to initialize LlamaScheduler.");
            return false;
        }

        _initialized = true;
        return true;
    } catch (const std::exception& e) {
        chorus_log(_log, LogLevel::Error, std::string("Exception during initialization: ") + e.what());
        return false;
    }
}

void LlamaEngine::submit_request(const ChorusRequest& chorus_request) {
    if (!_initialized || !scheduler) {
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
} // namespace Chorus