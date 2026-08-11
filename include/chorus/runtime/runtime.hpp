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
#include <variant>
#include <vector>

namespace Chorus {

// Terminal state of a session's most recent chat turn (spec #5-C). Protected
// metadata: read-only to hosts, never rendered into the prompt.
enum class TurnOutcome { None, Completed, Cancelled, Errored };

// What a host hands the runtime. No id, no callback: those are runtime business.
struct GenerationRequest {
    std::string prompt;

    // Caller-owned continuity lane (NPC, dialogue thread). Absent = stateless.
    // Empty string is InvalidRequest -- statelessness has one spelling.
    std::optional<SessionId> session_id;

    int priority = 0;
    bool stream = false; // false: suppress Token events, deliver only Complete

    // An overlay on the runtime's host defaults, not a finished config. Keeping
    // it a patch is what lets the runtime tell a value the caller asked for
    // from one that merely drifted down from a host's ambient settings.
    GenerationConfigPatch overrides;

    // #5: per-request ephemeral injections (fitted copy only) and an optional
    // chat-template override. Only meaningful on sessioned (chat) requests.
    std::vector<InjectedMessage> inject;
    std::string chat_template;
};

// A host's ambient settings: the layer every request overlays.
//
// Chat-only controls here (thinking, chat_template) apply to sessioned
// requests and are dropped from stateless ones. That asymmetry is the point:
// an ambient default must not turn a raw-prompt call into a rejection, while
// the same control set deliberately on a request still earns one. Resolving it
// here rather than in an adapter is what keeps every host from re-deriving the
// same workaround.
struct HostDefaults {
    GenerationConfigPatch config;
    std::string chat_template;
};

// What poll() returns. Post-policy, host-thread-safe.
struct RuntimeEvent {
    // EngineFailed reports an engine that died on its own rather than being
    // unloaded: request_id is -1, session_id absent, error EngineNotReady with
    // a generic message (the specific cause reaches the host through each dying
    // request's own Error terminal). It is not a "nothing further" marker.
    // Requests the engine had already queued may still terminate on a later
    // poll, so a host tears down on each request's own terminal, exactly as it
    // otherwise would.
    enum class Kind { Token, ReasoningToken, Complete, Error, HistoryTruncated, EngineFailed };
    RequestId request_id;
    std::optional<SessionId> session_id; // absent for stateless requests
    Kind kind;
    // A Token carries a safe text chunk and need not correspond to exactly one model token.
    std::string text; // Complete: full accumulated text; Error: message
    ChorusError error = ChorusError::None;
    std::string reasoning; // Complete: full accumulated reasoning ("" = none)
    int32_t dropped = 0;   // HistoryTruncated: history messages dropped
};

// Synchronous submit outcome. error != None means no request was created and
// request_id is -1.
struct SubmitResult {
    RequestId request_id = -1;
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

    // Installs the ambient layer every subsequent request overlays. Hosts that
    // expose node- or project-level settings push them here instead of folding
    // them into each request, so the runtime can distinguish the two.
    void set_host_defaults(HostDefaults defaults);

    [[nodiscard]] SubmitResult submit(const GenerationRequest& request);

    // Reroll the session's last assistant line (#5-C). request.prompt must be
    // empty; all other request fields (priority, stream, config, inject,
    // chat_template) apply to the rerolled turn. On Complete the new reply
    // replaces the old; on Cancelled/Error the old reply is restored.
    [[nodiscard]] SubmitResult regenerate(const GenerationRequest& request);

    // Requests stay active until poll() drains their terminal event.
    bool cancel(RequestId id);
    bool is_request_active(RequestId id) const;
    std::optional<RequestId> active_request_for_session(const SessionId& session_id) const;

    // --- Conversation history (#5). Host-thread-only, like everything else.
    // import/clear on a busy session (and reset with ANY busy session) return
    // SessionBusy: mutating a lane mid-turn would desync terminal rollback.
    std::optional<ChorusError> import_conversation_history(const SessionId& session, std::vector<ChatMessage> history);
    std::vector<ChatMessage> export_conversation_history(const SessionId& session) const;
    std::optional<ChorusError> clear_conversation_history(const SessionId& session);
    // Rewrites one message's content in place, any role. A negative index
    // counts from the end (-1 = newest). InvalidRequest for an unknown session
    // or an out-of-range index; SessionBusy while the lane has a live request.
    // The lane's last_turn_outcome is untouched: editing what a turn said does
    // not change how it ended.
    std::optional<ChorusError> edit_message(const SessionId& session, int64_t index, std::string content);
    std::vector<SessionId> list_conversations() const;
    std::optional<ChorusError> reset_context();
    TurnOutcome last_turn_outcome(const SessionId& session) const;

    // The exact fitted prompt generation would consume for this session right
    // now, without generating. Pass the SAME overrides you generate with -- the
    // host defaults apply underneath either way, and the resolved max_tokens
    // and thinking drive the fitting reservation.
    // nullopt: unknown session, no engine, or no provider rendering.
    std::optional<std::string> render_prompt(
        const SessionId& session,
        const std::string& template_override = "",
        const std::vector<InjectedMessage>& inject = {},
        const GenerationConfigPatch& overrides = {}
    ) const;

    // Drains pending engine signals into host-facing events.
    std::vector<RuntimeEvent> poll();

    // Drains buffered log records. Host-thread-only, like poll(); call it
    // beside poll(). Logs ride their own channel because log and token volumes
    // differ by orders of magnitude, so ordering between this stream and
    // poll()'s is explicitly not guaranteed. After a loss, the next batch
    // begins with one loss report as metadata; surviving records remain FIFO.
    std::vector<LogRecord> poll_logs();

    // Stops and destroys the engine. Every live request receives exactly one
    // Cancelled terminal on a later poll(). is_loaded() is false afterward.
    void stop_all();

  private:
    void assert_host_thread() const;
    void enqueue_signal(const ChorusSignal& signal);
    void unload_engine();
    void cancel_live_requests();
    void retire_request(RequestId id);

    // The EngineNotReady rejection every entry point shares, worded for which
    // kind of not-ready it is: no engine at all, or one that has failed.
    SubmitResult not_ready() const;

    // A request with its layers already collapsed: host defaults overlaid by
    // the request's own overrides, with ambient chat controls dropped where
    // they cannot apply. Everything below submit() works on this, so the
    // resolution rule has exactly one home.
    struct ResolvedRequest {
        const GenerationRequest& request;
        GenerationConfig config;
        std::string chat_template;
    };
    ResolvedRequest resolve_request(const GenerationRequest& request) const;
    static ChorusRequest make_engine_request(const ResolvedRequest& resolved);

    struct LiveRequest; // fwd for the chat-turn helpers below
    // Shared submit tail: id assignment, validation, live-state install,
    // durable history commit, dispatch. dropped > 0 queues a HistoryTruncated
    // host event for the next poll(). replaced_reply set = regenerate turn:
    // the commit pops the trailing assistant reply instead of appending the
    // user message, and the popped reply rides the live record for
    // restore-on-cancel.
    [[nodiscard]] SubmitResult submit_engine_request(
        const ResolvedRequest& resolved,
        ChorusRequest engine_request,
        int32_t dropped = 0,
        std::optional<ChatMessage> replaced_reply = std::nullopt
    );
    // Applies a chat turn's terminal to its session lane (append or rollback).
    void finish_turn(const LiveRequest& live, TurnOutcome outcome, const std::string& text);

    // A chat turn's message list after budget fitting (spec #5-F), or the
    // rejection to surface. Keeps prompt_fitting types out of this public
    // header while sparing callers an out-parameter.
    struct FittedTurn {
        std::vector<ChatMessage> messages;
        int32_t dropped = 0;
        // The probe's render of `messages`, captured during fitting (absent
        // when fitting was skipped): lets render_prompt reuse the final
        // probe render instead of rendering the winning candidate twice.
        std::optional<std::string> rendered_text;
    };
    std::variant<FittedTurn, SubmitResult>
    fit_turn_messages(const ResolvedRequest& resolved, std::vector<ChatMessage> prospective) const;

    struct LiveRequest {
        bool streaming = false;
        std::string accumulated_text;
        std::optional<SessionId> session_id;
        // #5 chat-turn bookkeeping. A sessioned request IS a chat turn:
        // Complete appends the assistant reply, Cancelled/Error rolls back.
        std::string accumulated_reasoning;
        // Present iff this turn is a regenerate: the assistant reply this turn
        // replaced, restored on cancel/error instead of the pop-trailing-user
        // rollback.
        std::optional<ChatMessage> replaced_reply;
    };

    std::unique_ptr<InferenceEngine> _engine;

    // Outlives every engine, and the sinks handed out over it hold it alive on
    // their own: a provider thread that somehow survives its engine still has
    // somewhere to write instead of a dangling reference.
    std::shared_ptr<LogChannel> _log_channel = std::make_shared<LogChannel>();

    // Host-thread-only. Set once poll() has told the host that the current
    // engine died, so the report happens exactly once per engine instance.
    // Whether the engine is dead is never cached: that is asked of the engine.
    bool _engine_failure_reported = false;

    HostDefaults _host_defaults;

    std::atomic<RequestId> _next_request_id{0};

    // Written by engine threads via enqueue_signal, drained by poll().
    std::mutex _pending_mutex;
    std::vector<ChorusSignal> _pending_signals;

    // Host-thread-only. A request is live from submit until its terminal event
    // is drained; this map is the single source of truth for request liveness.
    std::unordered_map<RequestId, LiveRequest> _live_requests;

    // Reverse index for session lookup and exclusivity. A session is busy from
    // submit until its terminal event is DRAINED by poll() -- not merely
    // emitted. Visible consequence: after stop_all()/replacement, a same-frame
    // resubmission for that session gets SessionBusy until the next poll()
    // drains the synthesized Cancelled terminal. Deliberate: drain-time release
    // guarantees per-session event ordering (#5's history append relies on it).
    std::unordered_map<SessionId, RequestId> _request_by_session;

    // #5 conversation store (host-thread-only). Explicit-only lifecycle:
    // lanes live until clear_conversation_history() or reset_context().
    struct ConversationHistory {
        std::vector<ChatMessage> messages;
        TurnOutcome last_turn_outcome = TurnOutcome::None;
    };
    std::unordered_map<SessionId, ConversationHistory> _histories;

    // Host-side events (e.g. HistoryTruncated) queued at submit time and
    // drained by poll() ahead of engine signals.
    std::vector<RuntimeEvent> _host_events;

    // Latched on first public call. Unconditional so class layout is stable
    // across debug/release TUs; only the assert_host_thread() body (and its
    // cost) is gated behind NDEBUG.
    mutable std::atomic<std::thread::id> _host_thread{};
};

} // namespace Chorus
