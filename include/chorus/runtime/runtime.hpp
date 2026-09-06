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

enum class TurnOutcome { None, Completed, Cancelled, Errored };

struct InferenceRequest {
    std::string prompt;
    int priority = 0; // Higher values indicate higher scheduling priority.
};

struct EmbeddingRequest : InferenceRequest {};

/// Describes one stateless generation or sessioned chat turn.
struct GenerationRequest : InferenceRequest {
    // `std::nullopt` selects stateless generation. An empty `Chorus::SessionId` is invalid.
    std::optional<SessionId> session_id;

    // Whether to emit `Chorus::RuntimeEvent::Kind::StreamedToken` and
    // `Chorus::RuntimeEvent::Kind::StreamedReasoningToken` events.
    bool stream = false;

    // Per-request changes layered over `Chorus::HostDefaults::config`.
    GenerationConfigPatch overrides;

    // Ephemeral messages inserted into this turn without changing stored history.
    std::vector<InjectedMessage> inject;

    // Per-request chat template. An empty value inherits `Chorus::HostDefaults::chat_template`.
    std::string chat_template;
};

/*
 * Ambient generation settings supplied by a host.
 *
 * Changes apply only to requests submitted afterward. Chat controls apply
 * only to sessions; request-level chat controls still reject stateless requests.
 */
struct HostDefaults {
    GenerationConfigPatch config;

    // Host chat template. An empty value delegates template selection to the engine.
    std::string chat_template;
};

/// One event delivered on the host thread by `Chorus::ChorusRuntime::poll`.
struct RuntimeEvent {
    enum class Kind {
        StreamedToken,          // Incremental visible text, emitted only for streaming requests.
        StreamedReasoningToken, // Incremental reasoning text, emitted only for streaming requests.
        Complete,               // Terminal success with complete visible and reasoning output.
        Embedding,              // Terminal success carrying a normalized embedding vector.
        Error,                  // Terminal request failure.
        HistoryTruncated,       // Prompt fitting omitted stored history messages; stored history is unchanged.
        EngineFailed            // Engine-wide failure that does not terminate a request.
    };

    // Accepted request ID, or `-1` for `Chorus::RuntimeEvent::Kind::EngineFailed`.
    RequestId request_id;

    // Absent for stateless requests and `Chorus::RuntimeEvent::Kind::EngineFailed`.
    std::optional<SessionId> session_id;

    Kind kind;

    // Streamed chunk on token events, full visible output on completion, or failure diagnostic.
    std::string text;

    // `Chorus::ChorusError::None` except on failure events.
    // Engine-wide failure always carries `Chorus::ChorusError::EngineNotReady`.
    ChorusError error = ChorusError::None;

    // Complete reasoning output on `Chorus::RuntimeEvent::Kind::Complete`.
    std::string reasoning;

    // Normalized vector on `Chorus::RuntimeEvent::Kind::Embedding`.
    std::vector<float> embedding;

    // History messages omitted from the fitted prompt by a truncation event.
    int32_t dropped = 0;
};

/*
 * The immediate result of attempting to start work.
 *
 * A rejection creates no request and emits no events.
 */
struct SubmitResult {
    // Nonnegative for an accepted request; `-1` for rejection.
    RequestId request_id = -1;

    // `Chorus::ChorusError::None` on acceptance; the rejection reason otherwise.
    ChorusError error = ChorusError::None;

    // Empty on acceptance; supplied by the rejecting layer otherwise.
    std::string message;

    bool ok() const { return error == ChorusError::None; }
};

/*
 * Owns request lifecycle between hosts and an inference engine.
 *
 * Public methods are confined to the first calling thread. Worker callbacks
 * remain buffered until the host calls `Chorus::ChorusRuntime::poll`.
 */
class ChorusRuntime {
  public:
    ChorusRuntime() = default;
    ~ChorusRuntime();

    ChorusRuntime(const ChorusRuntime&) = delete;
    ChorusRuntime& operator=(const ChorusRuntime&) = delete;

    /*
     * Replaces the engine and owns its lifetime.
     *
     * Live requests receive `Chorus::ChorusError::Cancelled`. The old engine
     * is destroyed before initialization; failure leaves the runtime unloaded.
     *
     * Returns:
     *  - `std::nullopt`: the engine initialized successfully.
     *  - `Chorus::ChorusError`: initialization failed.
     *
     * Errors:
     *  - `Chorus::ChorusError::InvalidRequest`: the supplied engine pointer is null.
     *  - `Chorus::ChorusError::UnsupportedModelFormat`: the provider cannot read the artifact.
     *  - `Chorus::ChorusError::ModelLoad`: the weights failed to load.
     *  - `Chorus::ChorusError::ContextInit`: the inference context failed to initialize.
     *  - `Chorus::ChorusError::UnsupportedOption`: a load option is unsupported.
     *  - `Chorus::ChorusError::Unknown`: the provider could not classify the failure.
     */
    std::optional<ChorusError> load_engine(std::unique_ptr<InferenceEngine> engine, const ChorusConfig& config);

    // Whether an initialized engine is ready to accept work.
    bool is_loaded() const;

    // Copies the effective capabilities of the initialized engine.
    std::optional<EngineCapabilities> capabilities() const;

    // Replaces the ambient settings layered beneath future requests.
    void set_host_defaults(HostDefaults defaults);

    /*
     * Submits a stateless generation or sessioned chat request.
     *
     * Rejection creates no request and emits no events. Every accepted request
     * later emits exactly one terminal event from `Chorus::ChorusRuntime::poll`.
     *
     * Errors:
     *  - `Chorus::ChorusError::EngineNotReady`: no initialized engine can accept work.
     *  - `Chorus::ChorusError::InvalidRequest`: the request is invalid or cannot fit.
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     *  - `Chorus::ChorusError::UnsupportedFeature`: the provider lacks a requested capability.
     *  - `Chorus::ChorusError::UnsupportedOption`: the provider rejects a requested option.
     */
    [[nodiscard]] SubmitResult submit(const GenerationRequest& request);

    /// Submits a stateless embedding request.
    [[nodiscard]] SubmitResult submit(const EmbeddingRequest& request);

    /*
     * Regenerates the latest assistant reply in a session.
     *
     * The request must name a nonempty session whose history ends in an
     * assistant message, and `Chorus::GenerationRequest::prompt` must be empty.
     * Completion replaces the reply; cancellation or error restores it.
     *
     * Errors:
     *  - `Chorus::ChorusError::EngineNotReady`: no initialized engine can accept work.
     *  - `Chorus::ChorusError::InvalidRequest`: the request or stored conversation is invalid.
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     *  - `Chorus::ChorusError::UnsupportedFeature`: the provider lacks a requested capability.
     *  - `Chorus::ChorusError::UnsupportedOption`: the provider rejects a requested option.
     */
    [[nodiscard]] SubmitResult regenerate(const GenerationRequest& request);

    /*
     * Requests cancellation.
     *
     * A successfully cancelled request remains active until
     * `Chorus::ChorusRuntime::poll` drains its
     * `Chorus::ChorusError::Cancelled` terminal event.
     *
     * Returns:
     *  - `true`: an active request has that ID and cancellation was requested.
     *  - `false`: no active request has that ID.
     */
    bool cancel(RequestId id);

    /// Whether a request with the supplied ID has a terminal event that remains undrained.
    bool is_request_active(RequestId id) const;

    /*
     * Finds the active request occupying one session.
     *
     * Returns:
     *  - `Chorus::RequestId`: the session's active request.
     *  - `std::nullopt`: the session has no active request.
     */
    std::optional<RequestId> active_request_for_session(const SessionId& session_id) const;

    /*
     * Lists every stored conversation, including an imported empty history.
     *
     * The result order is unspecified.
     */
    std::vector<SessionId> list_conversations() const;

    /// Copies stored history. Unknown and empty sessions both return an empty vector.
    std::vector<ChatMessage> export_conversation_history(const SessionId& session) const;

    /// Returns `Chorus::TurnOutcome::None` for an unknown session or one with no recorded turn.
    TurnOutcome last_turn_outcome(const SessionId& session) const;

    /*
     * Replaces one session's stored history and resets its recorded turn outcome.
     *
     * Returns:
     *  - `std::nullopt`: the history was imported.
     *  - `Chorus::ChorusError`: the history was rejected.
     *
     * Errors:
     *  - `Chorus::ChorusError::InvalidRequest`: the supplied session ID is empty.
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     */
    std::optional<ChorusError> import_conversation_history(const SessionId& session, std::vector<ChatMessage> history);

    /*
     * Rewrites one stored message without changing its recorded turn outcome.
     *
     * Negative indexes count backward from the newest message.
     *
     * Returns:
     *  - `std::nullopt`: the message was changed.
     *  - `Chorus::ChorusError`: the change was rejected.
     *
     * Errors:
     *  - `Chorus::ChorusError::InvalidRequest`: the session or index does not exist.
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     */
    std::optional<ChorusError> edit_message(const SessionId& session, int64_t index, std::string content);

    /*
     * Removes one session's stored history and turn outcome.
     *
     * Clearing an unknown session succeeds.
     *
     * Returns:
     *  - `std::nullopt`: the session is absent after the call.
     *  - `Chorus::ChorusError`: clearing was rejected.
     *
     * Errors:
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     */
    std::optional<ChorusError> clear_conversation_history(const SessionId& session);

    /*
     * Removes every stored conversation and turn outcome.
     *
     * Returns:
     *  - `std::nullopt`: all context was removed.
     *  - `Chorus::ChorusError`: reset was rejected.
     *
     * Errors:
     *  - `Chorus::ChorusError::SessionBusy`: at least one session has an active request.
     */
    std::optional<ChorusError> reset_context();

    /*
     * Renders the fitted prompt that generation would consume without submitting work.
     *
     * Returns:
     *  - `std::string`: the rendered prompt.
     *  - `std::nullopt`: the engine, stored session, prompt fit, or chat renderer is unavailable.
     */
    std::optional<std::string> render_prompt(
        const SessionId& session,
        const std::string& template_override = "",
        const std::vector<InjectedMessage>& inject = {},
        const GenerationConfigPatch& overrides = {}
    ) const;

    /// Drains buffered engine signals into host-facing events.
    std::vector<RuntimeEvent> poll();

    /*
     * Drains buffered log records.
     *
     * Log and inference events have independent FIFO channels and no ordering
     * guarantee between them. A batch following record loss begins with loss metadata.
     */
    std::vector<LogRecord> poll_logs();

    /// Stops and destroys the engine. Live requests receive one
    /// `Chorus::ChorusError::Cancelled` terminal event on a later `Chorus::ChorusRuntime::poll`.
    void stop_all();

  private:
    void assert_host_thread() const;
    void enqueue_signal(const ChorusSignal& signal);
    std::vector<ChorusSignal> drain_pending_signals();
    void append_signal_events(const ChorusSignal& signal, std::vector<RuntimeEvent>& events);
    void append_engine_failure(std::vector<RuntimeEvent>& events);
    void unload_engine();
    void cancel_live_requests();
    void retire_request(RequestId id);

    SubmitResult not_ready() const;

    struct ResolvedRequest {
        const GenerationRequest& request;
        GenerationConfig config;
        std::string chat_template;
    };
    ResolvedRequest resolve_request(const GenerationRequest& request) const;
    static ChorusRequest make_engine_request(const ResolvedRequest& resolved);

    struct LiveRequest;
    [[nodiscard]] SubmitResult submit_engine_request(
        const ResolvedRequest& resolved,
        ChorusRequest engine_request,
        int32_t dropped = 0,
        std::optional<ChatMessage> replaced_reply = std::nullopt
    );
    void finish_turn(const LiveRequest& live, TurnOutcome outcome, const std::string& text);

    struct FittedTurn {
        std::vector<ChatMessage> messages;
        int32_t dropped = 0;
        std::optional<std::string> rendered_text;
    };
    std::variant<FittedTurn, SubmitResult>
    fit_turn_messages(const ResolvedRequest& resolved, std::vector<ChatMessage> prospective) const;

    struct LiveRequest {
        RequestType type = RequestType::Generate;
        bool streaming = false;
        std::string accumulated_text;
        std::optional<SessionId> session_id;
        std::string accumulated_reasoning;
        std::optional<std::vector<float>> embedding;
        std::optional<ChatMessage> replaced_reply;
    };

    std::unique_ptr<InferenceEngine> _engine;

    std::shared_ptr<LogChannel> _log_channel = std::make_shared<LogChannel>();

    bool _engine_failure_reported = false;

    HostDefaults _host_defaults;

    RequestId _next_request_id = 0;

    std::mutex _pending_mutex;
    std::vector<ChorusSignal> _pending_signals;

    std::unordered_map<RequestId, LiveRequest> _live_requests;

    std::unordered_map<SessionId, RequestId> _request_by_session;

    struct ConversationHistory {
        std::vector<ChatMessage> messages;
        TurnOutcome last_turn_outcome = TurnOutcome::None;
    };
    std::unordered_map<SessionId, ConversationHistory> _histories;

    std::vector<RuntimeEvent> _host_events;

    mutable std::atomic<std::thread::id> _host_thread{};
};

} // namespace Chorus
