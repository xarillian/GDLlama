#include "chorus/providers/echo/echo_engine.hpp"

namespace Chorus {
EchoEngine::~EchoEngine() {
    shutdown();
}

std::optional<ChorusError> EchoEngine::initialize(const ChorusConfig& config, Logger logger) {
    _log = std::move(logger);

    if (_initialized) {
        _log.warn("Engine is already initialized");
        return std::nullopt;
    }

    // No model asset is needed; any InitialModelSpec is accepted and unread by design.
    if (!config.provider_options.empty()) {
        _log.error("Engine accepts no provider options");
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
        _log.for_request(chorus_request.id, chorus_request.session_id).error("Request submitted to a stopped engine");

        if (chorus_request.on_event) {
            ChorusSignal error_sig{
                chorus_request.id,
                ChorusSignal::Error{ChorusError::EngineNotReady, "Engine not initialized"},
            };
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
        ChorusSignal signal{
            queued->id,
            ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled."},
        };
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

void EchoEngine::shutdown() {
    std::vector<ChorusRequest> queued;
    {
        // The store must happen under the queue mutex: the worker evaluates its wait
        // predicate while holding it, and an unlocked store+notify can land between
        // that check and the block, losing the wakeup forever (shutdown() then hangs on
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
        ChorusSignal signal{
            request.id,
            ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled: engine stopped."},
        };
        request.on_event(signal);
    }
}

bool EchoEngine::is_initialized() const {
    return _initialized;
}

EngineCapabilities EchoEngine::capabilities() const {
    EngineCapabilities caps;
    caps.provider_id = "echo";
    caps.input_modalities = {Modality::Text};
    caps.output_modalities = {Modality::Text};
    caps.scheduling = SchedulingAuthority::ProviderManaged;
    caps.streaming = true;
    caps.cancellation = true;
    caps.common_generation_options = {"max_tokens"};
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
    const auto& c = request.gen_config;
    if (c.max_tokens && *c.max_tokens < -1)
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine max_tokens must be -1 or greater."};
    // Options addressed to this engine by namespace are demands: Echo has no
    // options, so any 'echo' entry is a typo to catch.
    if (request.gen_config.provider_options.count("echo"))
        return RequestRejection{
            ChorusError::UnsupportedOption, "EchoEngine has no provider options; remove the 'echo' entry."
        };

    // Content controls (sampling, stop, constraint, showing thinking, templates,
    // foreign provider namespaces) are inert here: echoed output makes no
    // content claims, so any value is vacuously honored. Accept them so real
    // request pipelines run unmodified against the test double, and warn once
    // per engine lifetime so the discard is not silent (user decision
    // 2026-07-17 amending the no-silent-discard posture for Echo).
    std::vector<const char*> ignored;
    if (c.temperature)
        ignored.push_back("temperature");
    if (c.top_k)
        ignored.push_back("top_k");
    if (c.top_p)
        ignored.push_back("top_p");
    if (c.seed)
        ignored.push_back("seed");
    if (c.frequency_penalty)
        ignored.push_back("frequency_penalty");
    if (c.presence_penalty)
        ignored.push_back("presence_penalty");
    if (!c.stop.empty())
        ignored.push_back("stop");
    if (c.constraint)
        ignored.push_back("constraint");
    if (c.show_thinking.has_value())
        ignored.push_back("show_thinking");
    if (!request.chat_template.empty())
        ignored.push_back("chat_template");
    if (!request.gen_config.provider_options.empty())
        ignored.push_back("provider_options");

    if (!ignored.empty() && !_warned_ignored.exchange(true)) {
        std::string names;
        for (size_t i = 0; i < ignored.size(); ++i) {
            if (i)
                names += ", ";
            names += ignored[i];
        }
        _log.warn("Ignoring content controls; echoed output makes no content claims", {{"controls", names}});
    }
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

        // Chat requests echo the last user message (keeps the model-free
        // path exercising the messages carrier).
        std::string chat_source;
        if (!req.messages.empty()) {
            chat_source = req.messages.back().content;
            for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
                if (it->role == "user") {
                    chat_source = it->content;
                    break;
                }
            }
        }
        // One Token per space-delimited chunk (spaces only, not all whitespace); each
        // chunk keeps its trailing space(s) so the concatenation of all token texts
        // equals the prompt exactly.
        const std::string& text = req.messages.empty() ? req.prompt : chat_source;
        size_t start = 0;
        int32_t chunks = 0;
        const int32_t max_chunks = req.gen_config.max_tokens.value_or(-1);
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

            ChorusSignal token_sig{
                req.id,
                ChorusSignal::Token{TokenChannel::Content, text.substr(start, end - start)},
            };
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

        ChorusSignal terminal =
            cancelled
                ? ChorusSignal{req.id, ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled."}}
                : ChorusSignal{req.id, ChorusSignal::Stop{}};
        req.on_event(terminal);
    }
}
} // namespace Chorus
