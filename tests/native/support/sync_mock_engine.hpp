#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"
#include "chorus/core/log.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Deterministic InferenceEngine for runtime tests: emits everything inline on
// the caller's thread (legal per the port contract). Knobs deliberately allow
// broken-provider behavior so the runtime's policy-boundary defenses can be
// exercised.
class SyncMockEngine : public Chorus::InferenceEngine {
  public:
    // --- behavior knobs ---
    std::vector<std::string> tokens{"Hello ", "world"};
    bool emit_duplicate_success = false;   // broken: second success terminal
    bool emit_token_after_success = false; // broken: trailing token after success
    int64_t rogue_extra_id = -1;           // broken: if >= 0, emit a token for this unknown id
    bool emit_wrong_kind_success = false;  // broken: replace success with the other request kind's terminal
    bool emit_token_on_embedding = false;  // broken: token before an embedding terminal
    Chorus::GenerationUsage completion_usage;
    std::vector<float> embedding_values{1.0F, 2.0F};
    Chorus::ChorusError fail_submit_with = Chorus::ChorusError::None; // inline error instead of tokens
    bool emit_error_instead_of_success = false;              // error instead of success, after any generation tokens
    std::optional<Chorus::ChorusError> fail_initialize_with; // make initialize() fail
    bool hold_requests = false;                              // accept but emit nothing (request stays in flight)
    bool emit_cancelled_on_cancel = false;                   // emit one Cancelled terminal for a matching held request
    bool emit_error_during_shutdown = false;                 // emit Error for held requests inside shutdown()
    bool log_on_initialize_from_worker = false;              // log from a thread the host does not own
    bool log_on_shutdown = false;                            // report during teardown, past any host's last poll
    bool supports_render = false;                            // render_chat_prompt returns text + word-count tokens
    std::optional<uint32_t> mock_per_request_context;        // reported via loaded_model_info()
    // When non-empty, submit_request emits exactly these channel-tagged tokens
    // (then the usual terminal) instead of `tokens`.
    std::vector<std::pair<Chorus::TokenChannel, std::string>> scripted_channel_tokens;

    // --- observability ---
    int initialize_calls = 0;
    int shutdown_calls = 0;
    std::string* seen_model_id = nullptr; // survives this object's destruction
    int* shutdown_count_sink = nullptr;   // ditto
    std::vector<int64_t> submitted_ids;
    std::vector<Chorus::RequestId> cancelled_ids;
    std::vector<Chorus::ChatMessage> last_messages; // messages of the last submitted request
    std::optional<std::string> last_chat_template;  // template of the last submitted request
    Chorus::GenerationConfig last_config;           // resolved config of the last submitted request
    Chorus::ExecutionMode last_execution = Chorus::ExecutionMode::Shared;

    std::optional<Chorus::InitializationFailure> initialize(
        const Chorus::ChorusConfig& config, Chorus::Logger logger, const Chorus::InitializationControl& control
    ) override {
        if (control.stop_token.stop_requested())
            return Chorus::InitializationFailure{Chorus::ChorusError::Cancelled, "mock initialization cancelled"};
        initialize_calls++;
        _log = std::move(logger);
        if (seen_model_id)
            *seen_model_id = config.model.model_id;
        if (log_on_initialize_from_worker) {
            std::thread([log = _log] { log.info("From worker"); }).join();
        }
        if (fail_initialize_with.has_value())
            return Chorus::InitializationFailure{*fail_initialize_with, "mock initialization failure"};
        _initialized = true;
        return std::nullopt;
    }

    bool is_initialized() const override { return _initialized; }

    Chorus::EngineCapabilities declared_caps = [] {
        Chorus::EngineCapabilities caps;
        caps.provider_id = "mock";
        caps.streaming = true;
        caps.embeddings = true;
        return caps;
    }();
    std::optional<Chorus::RequestRejection> reject_with; // validate_request returns this
    std::optional<Chorus::LoadedModelInfo> mock_model_info;

    Chorus::EngineCapabilities capabilities() const override {
        auto caps = declared_caps;
        caps.prompt_rendering = supports_render;
        caps.message_token_counting = supports_render;
        return caps;
    }
    std::optional<Chorus::LoadedModelInfo> loaded_model_info() const override {
        if (mock_per_request_context.has_value()) {
            Chorus::LoadedModelInfo info = mock_model_info.value_or(Chorus::LoadedModelInfo{});
            info.per_request_context = mock_per_request_context;
            return info;
        }
        return mock_model_info;
    }
    std::optional<Chorus::RenderedPrompt> render_chat_prompt(
        const std::vector<Chorus::ChatMessage>& messages, const std::optional<std::string>&, std::optional<bool>
    ) const override {
        return supports_render ? render(messages) : std::nullopt;
    }

    static std::optional<Chorus::RenderedPrompt> render(const std::vector<Chorus::ChatMessage>& messages) {
        Chorus::RenderedPrompt rendered;
        int32_t count = 0;
        for (const auto& message : messages) {
            const auto role = Chorus::message_role_name(message.role);
            const auto content = Chorus::joined_text(message.content);
            if (!role || !content)
                return std::nullopt;
            rendered.text += "<" + std::string(*role) + ">" + *content + "\n";
            count += 1; // per-message overhead
            bool in_word = false;
            for (char c : *content) {
                if (c == ' ') {
                    in_word = false;
                } else if (!in_word) {
                    in_word = true;
                    ++count;
                }
            }
        }
        rendered.token_count = count;
        return rendered;
    }

    std::optional<Chorus::RequestRejection> validate_request(const Chorus::ChorusRequest&) const override {
        return reject_with;
    }

    struct Preparation : Chorus::RequestPreparation {
        std::atomic<bool> closed{false};
        bool rendering = false;
        mutable std::mutex mutex;
        std::optional<Chorus::RequestRejection> rejection;
        std::optional<Chorus::RequestRejection> validate_request(const Chorus::ChorusRequest&) const override {
            std::lock_guard<std::mutex> lock(mutex);
            if (closed)
                return Chorus::RequestRejection{Chorus::ChorusError::EngineNotReady, "mock closed"};
            return rejection;
        }
        std::variant<Chorus::RenderedPrompt, Chorus::RequestRejection> render_chat_prompt(
            const std::vector<Chorus::ChatMessage>& messages, const std::optional<std::string>&, std::optional<bool>
        ) const override {
            if (closed)
                return Chorus::RequestRejection{Chorus::ChorusError::EngineNotReady, "mock closed"};
            if (rendering) {
                auto value = SyncMockEngine::render(messages);
                if (value)
                    return std::move(*value);
            }
            return Chorus::RequestRejection{Chorus::ChorusError::InvalidRequest, "mock render failed"};
        }
        std::variant<int64_t, Chorus::RequestRejection> count_message_tokens(const std::string& text) const override {
            if (closed)
                return Chorus::RequestRejection{Chorus::ChorusError::EngineNotReady, "mock closed"};
            int64_t count = 0;
            bool word = false;
            for (char c : text) {
                if (c == ' ')
                    word = false;
                else if (!word) {
                    word = true;
                    ++count;
                }
            }
            return count;
        }
    };
    std::shared_ptr<Chorus::RequestPreparation> request_preparation() const override {
        auto value = std::make_shared<Preparation>();
        value->rendering = supports_render;
        value->rejection = reject_with;
        _preparation = value;
        return value;
    }
    void set_rejection(std::optional<Chorus::RequestRejection> value) {
        reject_with = value;
        if (_preparation) {
            std::lock_guard<std::mutex> lock(_preparation->mutex);
            _preparation->rejection = std::move(value);
        }
    }

    void submit_request(Chorus::ChorusRequest req) override {
        submitted_ids.push_back(req.id);
        last_messages = req.messages;
        last_chat_template = req.chat_template;
        last_config = req.gen_config;
        last_execution = req.execution;
        if (!req.on_event)
            return;

        if (fail_submit_with != Chorus::ChorusError::None) {
            send(req.on_event, req.id, Chorus::ChorusSignal::Error{fail_submit_with, "mock failure"});
            return;
        }
        if (hold_requests) {
            _held.push_back(req);
            return;
        }

        if (req.type == Chorus::RequestType::Embedding) {
            if (emit_token_on_embedding)
                send(req.on_event, req.id, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, "wrong kind"});
            emit_terminal(req);
            return;
        }

        if (!scripted_channel_tokens.empty()) {
            for (const auto& [channel, text] : scripted_channel_tokens)
                send(req.on_event, req.id, Chorus::ChorusSignal::Token{channel, text});
        } else if (!tokens.empty()) {
            for (const auto& text : tokens)
                send(req.on_event, req.id, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, text});
        } else if (!req.messages.empty()) {
            // Deterministic assistant reply for chat-shaped tests: the last
            // user message's content, echoed as a single Token. Opt in by
            // setting `tokens = {}` (a non-empty `tokens` knob always wins).
            std::string reply;
            for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
                if (it->role == Chorus::MessageRole::User) {
                    reply = *Chorus::joined_text(it->content);
                    break;
                }
            }
            send(req.on_event, req.id, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, std::move(reply)});
        }
        if (rogue_extra_id >= 0)
            send(req.on_event, rogue_extra_id, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, "rogue"});
        emit_terminal(req);
    }

    void cancel_request(Chorus::RequestId id) override {
        cancelled_ids.push_back(id);
        if (!emit_cancelled_on_cancel)
            return;

        auto held = std::find_if(_held.begin(), _held.end(), [id](const auto& request) { return request.id == id; });
        if (held == _held.end())
            return;
        if (held->on_event)
            send(
                held->on_event,
                held->id,
                Chorus::ChorusSignal::Error{Chorus::ChorusError::Cancelled, "Request cancelled."}
            );
        _held.erase(held);
    }

    void shutdown() override {
        if (_preparation)
            _preparation->closed = true;
        shutdown_calls++;
        if (shutdown_count_sink)
            (*shutdown_count_sink)++;
        if (log_on_shutdown) {
            _log.warn("Shutting down with work outstanding");
            _log.info("Releasing weights");
        }
        if (emit_error_during_shutdown) {
            for (auto& req : _held)
                if (req.on_event)
                    send(
                        req.on_event,
                        req.id,
                        Chorus::ChorusSignal::Error{Chorus::ChorusError::Decode, "stopped mid-flight"}
                    );
        }
        _held.clear();
        _initialized = false;
    }

    /*
     * Dies mid-flight, the way the llama worker does on a fatal decode.
     *
     * Held requests get their Decode terminals first, then the engine stops
     * being able to take work. shutdown() is never called, so nothing fences the
     * callbacks and nothing announces the death: it is visible only to whoever
     * asks is_initialized() next.
     */
    void die() {
        for (auto& req : _held)
            if (req.on_event)
                send(
                    req.on_event,
                    req.id,
                    Chorus::ChorusSignal::Error{Chorus::ChorusError::Decode, "Inference decode failed."}
                );
        _held.clear();
        _initialized = false;
    }

  private:
    void emit_terminal(const Chorus::ChorusRequest& req) const {
        if (emit_error_instead_of_success) {
            send(
                req.on_event,
                req.id,
                Chorus::ChorusSignal::Error{
                    Chorus::ChorusError::Decode,
                    req.type == Chorus::RequestType::Embedding ? "embedding failed" : "failed after partial output"
                }
            );
            return;
        }
        const bool embedding = (req.type == Chorus::RequestType::Embedding) != emit_wrong_kind_success;
        const Chorus::ChorusSignal::Event success =
            embedding ? Chorus::ChorusSignal::Event{Chorus::ChorusSignal::Embedding{embedding_values}}
                      : Chorus::ChorusSignal::Event{Chorus::ChorusSignal::Completion{completion_usage}};
        send(req.on_event, req.id, success);
        if (emit_duplicate_success)
            send(req.on_event, req.id, success);
        if (emit_token_after_success)
            send(req.on_event, req.id, Chorus::ChorusSignal::Token{Chorus::TokenChannel::Content, "late"});
    }

    static void send(
        const std::function<void(Chorus::ChorusSignal&)>& callback,
        Chorus::RequestId id,
        Chorus::ChorusSignal::Event event
    ) {
        Chorus::ChorusSignal signal{id, std::move(event)};
        callback(signal);
    }

    mutable std::shared_ptr<Preparation> _preparation;
    bool _initialized = false;
    Chorus::Logger _log;
    std::vector<Chorus::ChorusRequest> _held;
};
