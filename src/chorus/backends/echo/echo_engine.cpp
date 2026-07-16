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
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        if (_initialized && _running) {
            _queue.push_back(chorus_request);
            accepted = true;
        }
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
        return;
    }

    _queue_cv.notify_one();
}

void EchoEngine::cancel_request(RequestId id) {
    std::optional<ChorusRequest> queued;
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        if (_active && _active->id == id) {
            _cancelled_ids.insert(id);
            return;
        }

        for (auto it = _queue.begin(); it != _queue.end(); ++it) {
            if (it->id != id)
                continue;
            queued = std::move(*it);
            _queue.erase(it);
            if (queued->on_event)
                ++_queued_cancel_callbacks_in_flight;
            break;
        }
    }

    if (queued && queued->on_event) {
        ChorusSignal signal;
        signal.request_id = queued->id;
        signal.type = EventType::Error;
        signal.error_code = ChorusError::Cancelled;
        signal.text = "Request cancelled.";
        try {
            queued->on_event(signal);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                --_queued_cancel_callbacks_in_flight;
            }
            _queue_cv.notify_all();
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            --_queued_cancel_callbacks_in_flight;
        }
        _queue_cv.notify_all();
    }
}

void EchoEngine::stop() {
    std::vector<ChorusRequest> queued;
    {
        // The store must happen under the queue mutex: the worker evaluates its wait
        // predicate while holding it, and an unlocked store+notify can land between
        // that check and the block, losing the wakeup forever (stop() then hangs on
        // join). The running flag alone does not prevent the lost wakeup.
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _running = false;
        if (_active)
            _cancelled_ids.insert(_active->id);
        while (!_queue.empty()) {
            queued.push_back(std::move(_queue.front()));
            _queue.pop_front();
        }
    }
    _queue_cv.notify_all();

    if (_worker.joinable()) {
        _worker.join();
    }
    {
        std::unique_lock<std::mutex> lock(_queue_mutex);
        _queue_cv.wait(lock, [this] { return _queued_cancel_callbacks_in_flight == 0; });
    }
    _initialized = false;

    for (auto& request : queued) {
        if (!request.on_event)
            continue;
        ChorusSignal signal;
        signal.request_id = request.id;
        signal.type = EventType::Error;
        signal.error_code = ChorusError::Cancelled;
        signal.text = "Request cancelled: engine stopped.";
        request.on_event(signal);
    }
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
    caps.cancellation = true;
    caps.portable_generation_options = {"max_tokens"};
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
    if (c.max_tokens && *c.max_tokens < -1)
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine max_tokens must be -1 or greater."};
    if (c.temperature || c.top_k || c.top_p || c.seed || c.frequency_penalty || c.presence_penalty || !c.stop.empty())
        return RequestRejection{
            ChorusError::UnsupportedOption,
            "EchoEngine honors only max_tokens; unset other options or use another backend."
        };
    if (!request.gen_config.backend_options.empty())
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine accepts no backend options."};
    return std::nullopt;
}

void EchoEngine::worker_loop() {
    while (true) {
        Chorus::ChorusRequest req;
        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            _queue_cv.wait(lock, [this] { return !_running || !_queue.empty(); });
            if (!_running)
                return;
            req = std::move(_queue.front());
            _queue.pop_front();
            _active = req;
        }

        if (!req.on_event) {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            _cancelled_ids.erase(req.id);
            _active.reset();
            continue;
        }

        // One Token per space-delimited chunk (spaces only, not all whitespace); each
        // chunk keeps its trailing space(s) so the concatenation of all token texts
        // equals the prompt exactly.
        const std::string& text = req.prompt;
        size_t start = 0;
        int32_t chunks = 0;
        const int32_t max_chunks = req.gen_config.common.max_tokens.value_or(-1);
        while (start < text.size() && (max_chunks < 0 || chunks < max_chunks)) {
            {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                if (_cancelled_ids.count(req.id) || !_running)
                    break;
            }

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
            ++chunks;
        }

        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            cancelled = _cancelled_ids.erase(req.id) > 0 || !_running;
            _active.reset();
        }

        ChorusSignal terminal;
        terminal.request_id = req.id;
        if (cancelled) {
            terminal.type = EventType::Error;
            terminal.error_code = ChorusError::Cancelled;
            terminal.text = "Request cancelled.";
        } else {
            terminal.type = EventType::Stop;
        }
        req.on_event(terminal);
    }
}
} // namespace Chorus
