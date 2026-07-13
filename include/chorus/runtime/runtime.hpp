#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Chorus {

// What a host hands the runtime. No id, no callback: those are runtime business.
struct GenerationRequest {
    std::string prompt;

    // Caller-owned continuity lane (NPC, dialogue thread). Absent = stateless.
    // Empty string is InvalidRequest -- statelessness has one spelling.
    std::optional<SessionId> session_id;

    int priority = 0;
    bool stream = false; // false: suppress Token events, deliver only Complete
    GenerationConfig config;
};

// What poll() returns. Post-policy, host-thread-safe.
struct RuntimeEvent {
    enum class Kind { Token, Complete, Error };
    int64_t request_id;
    std::optional<SessionId> session_id; // absent for stateless requests
    Kind kind;
    std::string text; // Token: the token; Complete: full accumulated text; Error: message
    ChorusError error = ChorusError::None;
};

// Synchronous submit outcome. error != None means no request was created and
// request_id is -1.
struct SubmitResult {
    int64_t request_id = -1;
    ChorusError error = ChorusError::None;
    std::string message; // Human-readable rejection detail; empty on success.
    bool ok() const { return error == ChorusError::None; }
};

// Application layer: owns the request lifecycle between hosts and engines.
//
// Threading: every public method is host-thread-only (latched on first call,
// asserted in debug builds). The engine's worker touches only the internal
// pending queue, through the on_event callback the runtime attaches. Pending
// events are retained without bound until drained: call poll() once per tick.
class ChorusRuntime {
  public:
    ChorusRuntime() = default;
    ~ChorusRuntime();

    ChorusRuntime(const ChorusRuntime&) = delete;
    ChorusRuntime& operator=(const ChorusRuntime&) = delete;

    // Takes ownership unconditionally. Replaces any current engine: live
    // requests receive a Cancelled terminal and the old engine is stopped and
    // destroyed BEFORE the new one initializes (no transient double model
    // residency). On failure the runtime is unloaded and the engine destroyed.
    std::optional<ChorusError> load_engine(std::unique_ptr<InferenceEngine> engine, const ChorusConfig& config);

    bool is_loaded() const;

    [[nodiscard]] SubmitResult submit(const GenerationRequest& request);

    // Drains pending engine signals into host-facing events.
    std::vector<RuntimeEvent> poll();

    // Stops and destroys the engine. Every live request receives exactly one
    // Cancelled terminal on a later poll(). is_loaded() is false afterward.
    void stop_all();

  private:
    void assert_host_thread();
    void enqueue_signal(const ChorusSignal& signal);
    void unload_engine();
    void cancel_live_requests();

    std::unique_ptr<InferenceEngine> _engine;

    std::atomic<int64_t> _next_request_id{0};

    // Written by engine threads via enqueue_signal, drained by poll().
    std::mutex _pending_mutex;
    std::vector<ChorusSignal> _pending_signals;

    // Host-thread-only. A request is "live" from submit until its terminal
    // event is drained; _request_streaming's key set defines liveness.
    std::unordered_map<int64_t, bool> _request_streaming;
    std::unordered_map<int64_t, std::string> _accumulator;

    // Host-thread-only session tracking. A session is busy from submit until
    // its terminal event is DRAINED by poll() -- not merely emitted. Visible
    // consequence: after stop_all()/replacement, a same-frame resubmission for
    // that session gets SessionBusy until the next poll() drains the
    // synthesized Cancelled terminal. Deliberate: drain-time release
    // guarantees per-session event ordering (#5's history append relies on it).
    std::unordered_map<int64_t, std::string> _request_sessions; // only sessioned requests
    std::unordered_set<std::string> _active_sessions;

    // Latched on first public call. Unconditional so class layout is stable
    // across debug/release TUs; only the assert_host_thread() body (and its
    // cost) is gated behind NDEBUG.
    std::atomic<std::thread::id> _host_thread{};
};

} // namespace Chorus
