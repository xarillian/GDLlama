#include "chorus/backends/echo/echo_engine.hpp"

namespace Chorus {
EchoEngine::EchoEngine() {}

EchoEngine::~EchoEngine() {
    stop();
}

std::optional<ChorusError> EchoEngine::initialize(const ChorusConfig& config) {
    _log = config.log_callback;

    if (_initialized) {
        chorus_log(_log, LogLevel::Warn, "EchoEngine is already initialized.");
        return std::nullopt;
    }

    // No model asset is needed; any ModelSpec is accepted and unread by design.
    if (!config.backend_options.empty()) {
        chorus_log(_log, LogLevel::Error, "EchoEngine accepts no backend options.");
        return ChorusError::UnsupportedOption;
    }

    _running = true;
    _worker = std::thread(&EchoEngine::worker_loop, this);
    _initialized = true;
    return std::nullopt;
}

void EchoEngine::submit_request(const ChorusRequest& chorus_request) {
    if (!_initialized) {
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

    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _queue.push(chorus_request);
    }
    _queue_cv.notify_one();
}

void EchoEngine::stop() {
    {
        // The store must happen under the queue mutex: the worker evaluates its wait
        // predicate while holding it, and an unlocked store+notify can land between
        // that check and the block, losing the wakeup forever (stop() then hangs on
        // join). Atomicity of _running alone does not prevent the lost wakeup.
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _running = false;
    }
    _queue_cv.notify_all();

    if (_worker.joinable()) {
        _worker.join();
    }
    _initialized = false;
}

bool EchoEngine::is_initialized() const {
    return _initialized;
}

EngineCapabilities EchoEngine::capabilities() const {
    EngineCapabilities caps;
    caps.backend_id = "echo";
    caps.input_modalities = {Modality::Text};
    caps.output_modalities = {Modality::Text};
    caps.scheduling = SchedulingAuthority::BackendManaged;
    caps.streaming = true;
    // Everything else stays false/empty: Echo honors no generation options
    // until #4 teaches it max_tokens and cancellation.
    return caps;
}

std::optional<LoadedModelInfo> EchoEngine::loaded_model_info() const {
    return std::nullopt; // model-free by design
}

std::optional<RequestRejection> EchoEngine::validate_request(const ChorusRequest& request) const {
    if (!_initialized)
        return RequestRejection{ChorusError::EngineNotReady, "EchoEngine is not initialized."};
    if (request.type == RequestType::Embedding)
        return RequestRejection{ChorusError::UnsupportedFeature, "EchoEngine does not produce embeddings."};
    if (request.gen_config.common.constraint)
        return RequestRejection{ChorusError::UnsupportedFeature, "EchoEngine supports no output constraints."};
    const auto& c = request.gen_config.common;
    if (c.max_tokens || c.temperature || c.top_k || c.top_p || c.seed || c.frequency_penalty || c.presence_penalty ||
        !c.stop.empty())
        return RequestRejection{
            ChorusError::UnsupportedOption,
            "EchoEngine honors no generation options; unset them or use another backend."
        };
    if (!request.gen_config.backend_options.empty())
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine accepts no backend options."};
    return std::nullopt;
}

void EchoEngine::worker_loop() {
    while (_running) {
        Chorus::ChorusRequest req;
        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            _queue_cv.wait(lock, [this] { return !_running || !_queue.empty(); });
            if (!_running)
                return;
            req = std::move(_queue.front());
            _queue.pop();
        }

        if (!req.on_event)
            continue;

        // One Token per space-delimited chunk (spaces only, not all whitespace); each
        // chunk keeps its trailing space(s) so the concatenation of all token texts
        // equals the prompt exactly.
        const std::string& text = req.prompt;
        size_t start = 0;
        while (start < text.size()) {
            size_t end = text.find(' ', start);
            if (end == std::string::npos) {
                end = text.size();
            } else {
                while (end < text.size() && text[end] == ' ')
                    ++end;
            }

            ChorusSignal token_sig;
            token_sig.request_id = req.id;
            token_sig.type = EventType::Token;
            token_sig.text = text.substr(start, end - start);
            req.on_event(token_sig);

            start = end;
        }

        ChorusSignal stop_sig;
        stop_sig.request_id = req.id;
        stop_sig.type = EventType::Stop;
        req.on_event(stop_sig);
    }
}
} // namespace Chorus
