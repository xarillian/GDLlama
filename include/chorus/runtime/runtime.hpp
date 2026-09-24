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

/*
 * A message spliced into a conversation at a fixed distance from its end.
 *
 * `InjectedMessage::depth == 0` indicates it should go after the last message.
 */
struct InjectedMessage {
    ChatMessage message;
    int32_t depth = 0;
};

struct ConversationMessage {
    MessageId id = -1;
    ChatMessage message;
};

struct InferenceRequest {
    std::string prompt;
    std::optional<SessionId> session_id;
    int priority = 0; // Higher values indicate higher scheduling priority.
    ExecutionMode execution = ExecutionMode::Shared;
};

struct EmbeddingRequest : InferenceRequest {};

/// Describes one stateless generation or sessioned chat turn.
struct GenerationRequest : InferenceRequest {
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
        EngineFailed,           // Engine-wide failure that does not terminate a request.
        PromptRendered,         // Terminal preview success, with text and omitted IDs.
        MessageTokenCount       // Terminal literal-content count success.
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

    // Present only on successful sessioned generation completion.
    std::optional<MessageId> message_id;

    // Stored history message identities omitted from the fitted prompt.
    std::vector<MessageId> omitted_message_ids;
    // Literal content count, valid only for `Chorus::RuntimeEvent::Kind::MessageTokenCount`.
    int64_t token_count = 0;
};

/*
 * The immediate result of attempting to start work.
 *
 * A rejection creates no request and emits no events.
 */
struct SubmitResult {
    // Nonnegative for an accepted request; `-1` for rejection.
    RequestId request_id = -1;

    // Present for an accepted new chat turn and its reserved reply respectively.
    std::optional<MessageId> request_message_id;
    std::optional<MessageId> response_message_id;

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
    ChorusRuntime();
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
    std::optional<InitializationFailure>
    load_engine(std::unique_ptr<InferenceEngine> engine, const ChorusConfig& config);

    // Whether an initialized engine is ready to accept work.
    bool is_loaded() const;

    std::optional<LoadedModelInfo> loaded_model_info() const;

    // Copies the effective capabilities of the initialized engine.
    std::optional<EngineCapabilities> capabilities() const;

    // Replaces the ambient settings layered beneath future requests.
    void set_host_defaults(HostDefaults defaults);

    /*
     * Submits a stateless generation or sessioned chat request.
     *
     * Rejection creates no request and emits no events. Every accepted request
     * later emits exactly one terminal event from `Chorus::ChorusRuntime::poll`.
     * Provider validation, tokenization and fitting happen after admission and
     * report failures asynchronously. Historical content is snapshotted without
     * copying its strings; caller-owned new input is copied for lifetime safety.
     *
     * Errors:
     *  - `Chorus::ChorusError::EngineNotReady`: no initialized engine can accept work.
     *  - `Chorus::ChorusError::InvalidRequest`: invalid shape or exhausted admission capacity.
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     *  - `Chorus::ChorusError::UnsupportedFeature`: the provider lacks a requested capability.
     *  - `Chorus::ChorusError::UnsupportedOption`: the provider rejects a requested option.
     */
    [[nodiscard]] SubmitResult submit(const GenerationRequest& request);

    /// Submits a stateless embedding request.
    [[nodiscard]] SubmitResult submit(const EmbeddingRequest& request);
    [[nodiscard]] std::vector<SubmitResult> submit_batch(const std::vector<GenerationRequest>& requests);
    [[nodiscard]] std::vector<SubmitResult> submit_batch(const std::vector<EmbeddingRequest>& requests);

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
    std::vector<ConversationMessage> export_conversation_history(const SessionId& session) const;

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
    std::optional<ChorusError>
    import_conversation_history(const SessionId& session, std::vector<ConversationMessage> history);

    /*
     * Rewrites one stored message identified by its nonnegative durable ID.
     *
     * Returns:
     *  - `std::nullopt`: the message was changed.
     *  - `Chorus::ChorusError`: the change was rejected.
     *
     * Errors:
     *  - `Chorus::ChorusError::InvalidRequest`: the session or message ID does not exist.
     *  - `Chorus::ChorusError::SessionBusy`: the session has an active request.
     */
    std::optional<ChorusError> edit_message(const SessionId& session, MessageId message_id, MessageContent content);

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
     * Accepts a read-only preview of a frozen history and defaults snapshot.
     *
     * Does not occupy a session or change its history or turn outcome. Success
     * arrives as `Chorus::RuntimeEvent::Kind::PromptRendered` from poll. Provider
     * validation and fitting errors arrive asynchronously after admission.
     * Deterministic templates select the same prompt as generation from the same
     * snapshot; a changed provider rerender can fail generation's final fit check.
     */
    [[nodiscard]] SubmitResult render_prompt(const GenerationRequest& request);

    /*
     * Accepts literal content counting without occupying a session.
     *
     * Parts are concatenated before tokenization, without automatic BOS/EOS or
     * special-token parsing. Counts exclude role/template/response reservation
     * and are not additive formatted costs. Success arrives as
     * `Chorus::RuntimeEvent::Kind::MessageTokenCount` from poll.
     */
    [[nodiscard]] SubmitResult count_message_tokens(MessageContent content);

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
    struct Control;
    struct PreparationJob;
    struct PreparationState;
    void enqueue_signal(const ChorusSignal& signal, const std::shared_ptr<Control>& control);
    void preparation_loop();
    void prepare(PreparationJob& job);
    void publish_error(RequestId id, const std::shared_ptr<Control>& control, ChorusError error, std::string message);
    void fail_preparation();
    void append_signal_events(const ChorusSignal& signal, std::vector<RuntimeEvent>& events);
    void append_engine_failure(std::vector<RuntimeEvent>& events);
    void unload_engine();
    void cancel_live_requests();
    void retire_request(RequestId id);

    SubmitResult not_ready() const;

    struct ResolvedRequest {
        GenerationRequest request;
        GenerationConfig config;
        std::string chat_template;
    };
    ResolvedRequest resolve_request(const GenerationRequest& request) const;
    static ChorusRequest make_engine_request(const ResolvedRequest& resolved);

    struct MessageNode {
        ConversationMessage value;
        uint64_t identity;
    };
    using MessageNodePtr = std::shared_ptr<const MessageNode>;
    using HistoryNodes = std::vector<MessageNodePtr>;
    using HistorySnapshot = std::shared_ptr<const HistoryNodes>;
    MessageNodePtr make_node(ConversationMessage message);

    enum class Operation { Generate, Embed, Preview, Count };
    struct LiveRequest;
    SubmitResult admit_generation(const GenerationRequest& request, Operation operation, bool regenerate = false);
    SubmitResult admit(std::unique_ptr<PreparationJob> job, MessageNodePtr replaced_reply = {});
    void finish_turn(const LiveRequest& live, TurnOutcome outcome, const std::string& text);
    void fit_turn_messages(PreparationJob& job);

    struct LiveRequest {
        Operation operation = Operation::Generate;
        std::shared_ptr<Control> control;
        bool streaming = false;
        std::string accumulated_text;
        std::optional<SessionId> session_id;
        std::string accumulated_reasoning;
        std::optional<std::vector<float>> embedding;
        std::optional<MessageId> pending_user_id;
        std::optional<MessageId> reserved_assistant_id;
        MessageNodePtr replaced_reply;
    };

    std::unique_ptr<InferenceEngine> _engine;

    std::shared_ptr<LogChannel> _log_channel = std::make_shared<LogChannel>();

    bool _engine_failure_reported = false;

    HostDefaults _host_defaults;

    RequestId _next_request_id = 0;
    uint64_t _next_content_identity = 0;
    std::unique_ptr<PreparationState> _preparation;

    std::unordered_map<RequestId, LiveRequest> _live_requests;

    std::unordered_map<SessionId, RequestId> _request_by_session;

    struct ConversationHistory {
        HistorySnapshot messages = std::make_shared<const HistoryNodes>();
        TurnOutcome last_turn_outcome = TurnOutcome::None;
    };
    std::unordered_map<SessionId, ConversationHistory> _histories;

    std::optional<MessageId> _next_message_id = 0;

    mutable std::atomic<std::thread::id> _host_thread{};
};

} // namespace Chorus
