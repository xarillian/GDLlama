#include "chorus/providers/echo/echo_engine.hpp"

#include <cmath>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <vector>

namespace Chorus {
namespace {
struct QueuedCancelCallbackFrame {
    EchoEngine* engine;
    QueuedCancelCallbackFrame* previous;
};

thread_local QueuedCancelCallbackFrame* current_queued_cancel_callback = nullptr;

uint32_t murmur3_32(std::string_view value, uint32_t seed = 0x9747b28cU) {
    uint32_t hash = seed;
    size_t offset = 0;
    while (offset + 4 <= value.size()) {
        uint32_t block = static_cast<uint8_t>(value[offset]) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(value[offset + 1])) << 8) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(value[offset + 2])) << 16) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(value[offset + 3])) << 24);
        block *= 0xcc9e2d51U;
        block = (block << 15) | (block >> 17);
        block *= 0x1b873593U;
        hash ^= block;
        hash = ((hash << 13) | (hash >> 19)) * 5U + 0xe6546b64U;
        offset += 4;
    }

    uint32_t tail = 0;
    switch (value.size() - offset) {
    case 3:
        tail ^= static_cast<uint32_t>(static_cast<uint8_t>(value[offset + 2])) << 16;
        [[fallthrough]];
    case 2:
        tail ^= static_cast<uint32_t>(static_cast<uint8_t>(value[offset + 1])) << 8;
        [[fallthrough]];
    case 1:
        tail ^= static_cast<uint8_t>(value[offset]);
        tail *= 0xcc9e2d51U;
        tail = (tail << 15) | (tail >> 17);
        tail *= 0x1b873593U;
        hash ^= tail;
        break;
    default:
        break;
    }
    hash ^= static_cast<uint32_t>(value.size());
    hash ^= hash >> 16;
    hash *= 0x85ebca6bU;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35U;
    hash ^= hash >> 16;
    return hash;
}

bool is_ascii_word(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

std::vector<float> make_echo_embedding(const std::string& prompt) {
    std::vector<std::string> words;
    for (size_t i = 0; i < prompt.size();) {
        while (i < prompt.size() && !is_ascii_word(static_cast<unsigned char>(prompt[i])))
            ++i;
        std::string word;
        while (i < prompt.size() && is_ascii_word(static_cast<unsigned char>(prompt[i]))) {
            unsigned char c = static_cast<unsigned char>(prompt[i++]);
            word.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : static_cast<char>(c));
        }
        if (!word.empty())
            words.push_back(std::move(word));
    }

    std::vector<float> values(128, 0.0F);
    const auto add_feature = [&values](std::string_view feature) {
        const uint32_t hash = murmur3_32(feature);
        values[hash & 127U] += (hash & 0x80000000U) ? -1.0F : 1.0F;
    };
    for (const auto& word : words)
        add_feature("u:" + word);
    for (size_t i = 1; i < words.size(); ++i)
        add_feature("b:" + words[i - 1] + "\x1f" + words[i]);
    if (words.empty())
        add_feature(std::string("r:") + prompt);

    double squared_norm = 0.0;
    for (float value : values)
        squared_norm += static_cast<double>(value) * value;
    const double norm = std::sqrt(squared_norm);
    for (float& value : values)
        value = static_cast<float>(value / norm);
    return values;
}
} // namespace

struct EchoEngine::Preparation : RequestPreparation {
    mutable std::mutex mutex;
    bool closed = false;

    std::optional<RequestRejection> validate_request(const ChorusRequest& request) const override;
    std::variant<RenderedPrompt, RequestRejection> render_chat_prompt(
        const std::vector<ChatMessage>&, const std::optional<std::string>&, std::optional<bool>
    ) const override {
        std::lock_guard<std::mutex> lock(mutex);
        return RequestRejection{
            closed ? ChorusError::EngineNotReady : ChorusError::UnsupportedFeature, "Echo does not render chat prompts."
        };
    }
    std::variant<int64_t, RequestRejection> count_message_tokens(const std::string&) const override {
        std::lock_guard<std::mutex> lock(mutex);
        return RequestRejection{
            closed ? ChorusError::EngineNotReady : ChorusError::UnsupportedFeature, "Echo has no tokenizer."
        };
    }
};

EchoEngine::~EchoEngine() {
    shutdown();
}

std::optional<InitializationFailure>
EchoEngine::initialize(const ChorusConfig& config, Logger logger, const InitializationControl& control) {
    _log = std::move(logger);

    if (_initialized) {
        _log.warn("Engine is already initialized");
        return std::nullopt;
    }

    // No model asset is needed; model configuration is accepted and unread by design.
    if (!config.provider_options.empty()) {
        _log.error("Engine accepts no provider options");
        return InitializationFailure{ChorusError::UnsupportedOption, "EchoEngine accepts no provider options."};
    }

    if (control.stop_token.stop_requested())
        return InitializationFailure{ChorusError::Cancelled, "Echo initialization cancelled."};
    if (control.on_progress)
        control.on_progress({LoadPhase::InitializingEngine, std::nullopt});
    try {
        _preparation = std::make_shared<Preparation>();
        if (control.stop_token.stop_requested()) {
            shutdown();
            return InitializationFailure{ChorusError::Cancelled, "Echo initialization cancelled."};
        }
        _running = true;
        _worker = std::thread(&EchoEngine::worker_loop, this);
        if (control.stop_token.stop_requested()) {
            shutdown();
            return InitializationFailure{ChorusError::Cancelled, "Echo initialization cancelled."};
        }
        _initialized = true;
        return std::nullopt;
    } catch (const std::exception& e) {
        shutdown();
        return InitializationFailure{ChorusError::Unknown, e.what()};
    } catch (...) {
        shutdown();
        return InitializationFailure{ChorusError::Unknown, "Echo initialization failed."};
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
    caps.embeddings = true;
    caps.common_generation_options = {"max_tokens"};
    return caps;
}

std::optional<LoadedModelInfo> EchoEngine::loaded_model_info() const {
    return std::nullopt;
}

std::shared_ptr<RequestPreparation> EchoEngine::request_preparation() const {
    return _preparation;
}

std::optional<RequestRejection> EchoEngine::validate_request(const ChorusRequest& request) const {
    if (!_preparation)
        return RequestRejection{ChorusError::EngineNotReady, "EchoEngine is not initialized."};
    return _preparation->validate_request(request);
}

std::optional<RequestRejection> EchoEngine::Preparation::validate_request(const ChorusRequest& request) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (closed)
        return RequestRejection{ChorusError::EngineNotReady, "EchoEngine is not initialized."};
    if (request.type == RequestType::Embedding) {
        if (request.prompt.empty())
            return RequestRejection{ChorusError::InvalidRequest, "Embedding prompts must not be empty."};
        return std::nullopt;
    }
    for (const auto& message : request.messages) {
        if (!message_role_name(message.role))
            return RequestRejection{ChorusError::InvalidRequest, "Chat message role is invalid."};
        if (!joined_text(message.content))
            return RequestRejection{ChorusError::UnsupportedFeature, "EchoEngine accepts text-only chat content."};
    }
    const auto& c = request.gen_config;
    if (c.max_tokens && *c.max_tokens < -1)
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine max_tokens must be -1 or greater."};
    if (c.temperature || c.top_k || c.top_p || c.seed || c.frequency_penalty || c.presence_penalty)
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine does not support sampling options."};
    if (c.stop && !c.stop->empty())
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine does not support stop markers."};
    if (c.constraint && !std::holds_alternative<UnconstrainedOutput>(*c.constraint))
        return RequestRejection{ChorusError::UnsupportedFeature, "EchoEngine does not support output constraints."};
    if (c.show_thinking == true)
        return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine does not support reasoning."};
    if (request.chat_template) {
        return RequestRejection{
            request.chat_template->empty() ? ChorusError::InvalidRequest : ChorusError::UnsupportedOption,
            request.chat_template->empty() ? "EchoEngine chat_template must not be empty."
                                           : "EchoEngine does not support chat templates."
        };
    }
    if (auto found = c.provider_options.find("echo"); found != c.provider_options.end()) {
        const auto* options = std::get_if<ProviderOptionMap>(&found->second);
        if (!options || !options->empty())
            return RequestRejection{ChorusError::UnsupportedOption, "EchoEngine has no provider options."};
    }
    return std::nullopt;
}

void EchoEngine::submit_request(ChorusRequest chorus_request) {
    std::unique_lock<std::mutex> lock(_queue_mutex);
    if (_initialized && _running) {
        _queue.push_back(std::move(chorus_request));
        lock.unlock();
        _queue_cv.notify_one();
        return;
    }
    lock.unlock();

    _log.for_request(chorus_request.id, chorus_request.session_id).error("Request submitted to a stopped engine");
    if (chorus_request.on_event) {
        ChorusSignal error_sig{
            chorus_request.id,
            ChorusSignal::Error{ChorusError::EngineNotReady, "Engine not initialized"},
        };
        chorus_request.on_event(error_sig);
    }
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
        QueuedCancelCallbackFrame frame{this, current_queued_cancel_callback};
        current_queued_cancel_callback = &frame;
        const auto finish_callback = [this, &frame] {
            current_queued_cancel_callback = frame.previous;
            {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                --_queued_cancel_callbacks_in_flight;
            }
            _queue_cv.notify_all();
        };
        try {
            queued->on_event(signal);
        } catch (...) {
            finish_callback();
            throw;
        }
        finish_callback();
    }
}

void EchoEngine::shutdown() {
    if (_preparation) {
        std::lock_guard<std::mutex> lock(_preparation->mutex);
        _preparation->closed = true;
    }
    size_t callbacks_on_this_thread = 0;
    for (auto* frame = current_queued_cancel_callback; frame; frame = frame->previous)
        callbacks_on_this_thread += frame->engine == this;

    std::vector<ChorusRequest> queued;
    {
        // Store under `_queue_mutex`: the worker evaluates its wait predicate while
        // holding the same mutex. An unlocked store and notification can land between
        // that check and the wait, losing the wakeup and leaving
        // `Chorus::EchoEngine::shutdown` blocked in `std::thread::join`.
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
        // The current callback cannot wait for its own return. Every callback
        // running on another thread remains inside the shutdown fence.
        _queue_cv.wait(lock, [this, callbacks_on_this_thread] {
            return _queued_cancel_callbacks_in_flight == callbacks_on_this_thread;
        });
    }
    _initialized = false;
    _preparation.reset();

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

        if (req.type == RequestType::Embedding) {
            bool cancelled = false;
            {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                cancelled = _cancelled_ids.count(req.id) > 0 || !_running;
            }
            if (!cancelled) {
                ChorusSignal embedding{req.id, ChorusSignal::Embedding{make_echo_embedding(req.prompt)}};
                req.on_event(embedding);
            }
        } else {
            const std::string text = select_echo_text(req);
            emit_echo_tokens(req, text);
        }

        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            cancelled = _cancelled_ids.erase(req.id) > 0 || !_running;
            _active.reset();
        }

        ChorusSignal terminal =
            cancelled ? ChorusSignal{req.id, ChorusSignal::Error{ChorusError::Cancelled, "Request cancelled."}}
                      : ChorusSignal{req.id, ChorusSignal::Stop{}};
        req.on_event(terminal);
    }
}

std::string EchoEngine::select_echo_text(const ChorusRequest& request) {
    if (request.messages.empty())
        return request.prompt;
    for (const auto& message : std::views::reverse(request.messages)) {
        if (message.role == MessageRole::User)
            return joined_text(message.content).value();
    }
    return joined_text(request.messages.back().content).value();
}

void EchoEngine::emit_echo_tokens(const ChorusRequest& request, const std::string& text) {
    // Split only on spaces and retain each run so concatenating the emitted
    // `Chorus::ChorusSignal::Token` content reconstructs the input exactly.
    size_t start = 0;
    int32_t chunks = 0;
    const int32_t max_chunks = request.gen_config.max_tokens.value_or(-1);
    while (start < text.size() && (max_chunks < 0 || chunks < max_chunks)) {
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            if (_cancelled_ids.count(request.id) || !_running)
                break;
        }

        size_t end = text.find(' ', start);
        if (end == std::string::npos) {
            end = text.size();
        } else {
            while (end < text.size() && text[end] == ' ')
                ++end;
        }

        ChorusSignal token_signal{
            request.id,
            ChorusSignal::Token{TokenChannel::Content, text.substr(start, end - start)},
        };
        request.on_event(token_signal);

        start = end;
        ++chunks;
    }
}
} // namespace Chorus
