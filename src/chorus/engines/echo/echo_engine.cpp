#include "chorus/engines/echo/echo_engine.hpp"

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

    // No model file needed; model_path is ignored by design.
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
