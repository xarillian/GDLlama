#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"
#include "chorus/core/log.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Deterministic InferenceEngine for runtime tests: emits everything inline on
// the caller's thread (legal per the port contract). Knobs deliberately allow
// broken-backend behavior so the runtime's policy-boundary defenses can be
// exercised.
class SyncMockEngine : public Chorus::InferenceEngine {
  public:
    // --- behavior knobs ---
    std::vector<std::string> tokens{"Hello ", "world"};
    bool emit_stop = true;              // emit Stop after tokens
    bool emit_duplicate_stop = false;   // broken: second Stop after the first
    bool emit_token_after_stop = false; // broken: trailing Token after Stop
    int64_t rogue_extra_id = -1;        // broken: if >= 0, emit a Token for this unknown id
    bool emit_embedding_event = false;  // pre-#6 kind the runtime must drop
    Chorus::ChorusError fail_submit_with = Chorus::ChorusError::None; // inline Error instead of tokens
    bool emit_error_instead_of_stop = false;                 // tokens flow, then Error terminal (partial-output shape)
    std::optional<Chorus::ChorusError> fail_initialize_with; // make initialize() fail
    bool hold_requests = false;                              // accept but emit nothing (request stays in flight)
    bool emit_cancelled_on_cancel = false;                   // emit one Cancelled terminal for a matching held request
    bool emit_error_during_stop = false;                     // emit Error for held requests inside stop()
    bool log_on_initialize_from_worker = false;              // exercise log-callback pass-through
    bool supports_render = false;                            // render_chat_prompt returns text + word-count tokens
    std::optional<uint32_t> mock_per_request_context;        // reported via loaded_model_info()
    // When non-empty, submit_request emits exactly these channel-tagged tokens
    // (then the usual terminal) instead of `tokens`.
    std::vector<std::pair<Chorus::TokenChannel, std::string>> scripted_channel_tokens;

    // --- observability ---
    int initialize_calls = 0;
    int stop_calls = 0;
    std::string* seen_model_id = nullptr; // survives this object's destruction
    int* stop_count_sink = nullptr;       // ditto
    std::vector<int64_t> submitted_ids;
    std::vector<Chorus::RequestId> cancelled_ids;
    std::vector<Chorus::ChatMessage> last_messages; // messages of the last submitted request
    std::string last_chat_template;                 // template of the last submitted request
    Chorus::GenerationConfig last_config;           // resolved config of the last submitted request

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config) override {
        initialize_calls++;
        if (seen_model_id)
            *seen_model_id = config.model.model_id;
        if (log_on_initialize_from_worker && config.log_callback) {
            std::thread([cb = config.log_callback] {
                Chorus::chorus_log(cb, Chorus::LogLevel::Info, "from worker");
            }).join();
        }
        if (fail_initialize_with.has_value())
            return fail_initialize_with;
        _initialized = true;
        return std::nullopt;
    }

    bool is_initialized() const override { return _initialized; }

    Chorus::EngineCapabilities declared_caps = [] {
        Chorus::EngineCapabilities caps;
        caps.backend_id = "mock";
        caps.streaming = true;
        return caps;
    }();
    std::optional<Chorus::RequestRejection> reject_with; // validate_request returns this
    std::optional<Chorus::LoadedModelInfo> mock_model_info;

    Chorus::EngineCapabilities capabilities() const override { return declared_caps; }
    std::optional<Chorus::LoadedModelInfo> loaded_model_info() const override {
        if (mock_per_request_context.has_value()) {
            Chorus::LoadedModelInfo info = mock_model_info.value_or(Chorus::LoadedModelInfo{});
            info.per_request_context = mock_per_request_context;
            return info;
        }
        return mock_model_info;
    }
    std::optional<Chorus::RequestRejection> validate_request(const Chorus::ChorusRequest&) const override {
        return reject_with;
    }

    std::optional<Chorus::RenderedPrompt> render_chat_prompt(
        const std::vector<Chorus::ChatMessage>& messages,
        const std::string& /*template_override*/,
        bool /*enable_thinking*/
    ) const override {
        if (!supports_render)
            return std::nullopt;
        Chorus::RenderedPrompt rendered;
        int32_t count = 0;
        for (const auto& message : messages) {
            rendered.text += "<" + message.role + ">" + message.content + "\n";
            count += 1; // per-message overhead
            bool in_word = false;
            for (char c : message.content) {
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

    void submit_request(const Chorus::ChorusRequest& req) override {
        submitted_ids.push_back(req.id);
        last_messages = req.messages;
        last_chat_template = req.chat_template;
        last_config = req.gen_config;
        if (!req.on_event)
            return;

        if (fail_submit_with != Chorus::ChorusError::None) {
            send(req.on_event, req.id, Chorus::EventType::Error, "mock failure", fail_submit_with);
            return;
        }
        if (hold_requests) {
            _held.push_back(req);
            return;
        }

        if (!scripted_channel_tokens.empty()) {
            for (const auto& [channel, text] : scripted_channel_tokens) {
                Chorus::ChorusSignal sig;
                sig.request_id = req.id;
                sig.type = Chorus::EventType::Token;
                sig.channel = channel;
                sig.text = text;
                req.on_event(sig);
            }
            if (emit_stop)
                send(req.on_event, req.id, Chorus::EventType::Stop, "");
            return;
        }

        if (!tokens.empty()) {
            for (const auto& t : tokens)
                send(req.on_event, req.id, Chorus::EventType::Token, t);
        } else if (!req.messages.empty()) {
            // Deterministic assistant reply for chat-shaped tests: the last
            // user message's content, echoed as a single Token. Opt in by
            // setting `tokens = {}` (a non-empty `tokens` knob always wins).
            std::string reply;
            for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
                if (it->role == "user") {
                    reply = it->content;
                    break;
                }
            }
            send(req.on_event, req.id, Chorus::EventType::Token, reply);
        }
        if (rogue_extra_id >= 0)
            send(req.on_event, rogue_extra_id, Chorus::EventType::Token, "rogue");
        if (emit_embedding_event)
            send(req.on_event, req.id, Chorus::EventType::Embedding, "");
        if (emit_error_instead_of_stop) {
            send(
                req.on_event,
                req.id,
                Chorus::EventType::Error,
                "failed after partial output",
                Chorus::ChorusError::Decode
            );
            return;
        }
        if (emit_stop)
            send(req.on_event, req.id, Chorus::EventType::Stop, "");
        if (emit_duplicate_stop)
            send(req.on_event, req.id, Chorus::EventType::Stop, "");
        if (emit_token_after_stop)
            send(req.on_event, req.id, Chorus::EventType::Token, "late");
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
                held->on_event, held->id, Chorus::EventType::Error, "Request cancelled.", Chorus::ChorusError::Cancelled
            );
        _held.erase(held);
    }

    void stop() override {
        stop_calls++;
        if (stop_count_sink)
            (*stop_count_sink)++;
        if (emit_error_during_stop) {
            for (auto& req : _held)
                if (req.on_event)
                    send(
                        req.on_event,
                        req.id,
                        Chorus::EventType::Error,
                        "stopped mid-flight",
                        Chorus::ChorusError::Decode
                    );
        }
        _held.clear();
        _initialized = false;
    }

  private:
    static void send(
        const std::function<void(Chorus::ChorusSignal&)>& cb,
        int64_t id,
        Chorus::EventType type,
        const std::string& text,
        Chorus::ChorusError err = Chorus::ChorusError::None
    ) {
        Chorus::ChorusSignal sig;
        sig.request_id = id;
        sig.type = type;
        sig.error_code = err;
        sig.text = text;
        cb(sig);
    }

    bool _initialized = false;
    std::vector<Chorus::ChorusRequest> _held;
};
