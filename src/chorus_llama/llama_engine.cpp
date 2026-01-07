#include "chorus_llama/llama_engine.hpp"
#include "chorus_llama/llama_scheduler.hpp"
#include "chorus_core/chorus_common.hpp"

#include <iostream>


namespace Chorus {
    LlamaEngine::LlamaEngine() {
        // constructor can be empty -- `initialize` does the work
    }

    LlamaEngine::~LlamaEngine() {
        stop();
    }

    bool LlamaEngine::initialize(const ChorusConfig& config) {
        if (_initialized) {
            std::cerr << "[Chorus] Warning: LlamaEngine is already initialized." << std::endl;
            return true;
        }

        if (config.model_path.empty()) {
            std::cerr << "[Chorus] Error: Model path is empty." << std::endl;
            return false;
        }

        try {
            scheduler = std::make_unique<LlamaScheduler>();

            if (!scheduler->initialize(config)) {
                scheduler.reset();
                std::cerr << "[Chorus] Error: Failed to initialize LlamaScheduler." << std::endl;
                return false;
            }

            _initialized = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Chorus] Exception during initialization: " << e.what() << std::endl;
            return false;
        }
    }

    void LlamaEngine::submit_request(const ChorusRequest& chorus_request) {
        if (!_initialized || !scheduler) {
            std::cerr << "[Chorus] Error: Attempting to submit request to uninitialized engine." << std::endl;

            if (chorus_request.on_event) {
                ChorusSignal error_sig;
                error_sig.request_id = chorus_request.id;
                error_sig.type = EventType::Error;
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
}