#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"
#include "chorus/core/log.hpp"

#include <functional>
#include <optional>
#include <string>
#include <thread>
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
    bool emit_error_during_stop = false;                     // emit Error for held requests inside stop()
    bool log_on_initialize_from_worker = false;              // exercise log-callback pass-through

    // --- observability ---
    int initialize_calls = 0;
    int stop_calls = 0;
    std::string* seen_model_id = nullptr; // survives this object's destruction
    int* stop_count_sink = nullptr;       // ditto
    std::vector<int64_t> submitted_ids;

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
    std::optional<Chorus::LoadedModelInfo> loaded_model_info() const override { return mock_model_info; }
    std::optional<Chorus::RequestRejection> validate_request(const Chorus::ChorusRequest&) const override {
        return reject_with;
    }

    void submit_request(const Chorus::ChorusRequest& req) override {
        submitted_ids.push_back(req.id);
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

        for (const auto& t : tokens)
            send(req.on_event, req.id, Chorus::EventType::Token, t);
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
