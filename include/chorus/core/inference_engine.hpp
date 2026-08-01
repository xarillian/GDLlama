#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <optional>

namespace Chorus {

/*
 * The provider port, the only way anything above reaches a provider.
 *
 * Every inference provider implements this interface. An engine may run
 * threads of its own, so every callback it is handed must be thread-safe. Its
 * own methods run the other way around, confined to the thread that calls them
 * and never reached from an engine worker.
 */
class InferenceEngine {
  public:
    virtual ~InferenceEngine() = default;

    /*
     * Brings the engine up under `chorus_config`.
     *
     * Initializing an engine that is already running does nothing and
     * succeeds. One that started and later died frees what it still holds
     * before retrying, so two models are never resident at once.
     *
     * Errors are returned instead of signalled.
     *
     * Returns:
     *  - `std::nullopt`: the engine came up.
     *  - `ChorusError`: what stopped it.
     *
     * Errors:
     *  - `ChorusError::UnsupportedModelFormat`: the provider cannot read the artifact.
     *  - `ChorusError::ModelLoad`: the weights will not load.
     *  - `ChorusError::ContextInit`: the inference context will not build.
     *  - `ChorusError::UnsupportedOption`: a load option the provider does not declare.
     *  - `ChorusError::Unknown`: the provider failed with nothing better to say.
     */
    virtual std::optional<ChorusError> initialize(const Chorus::ChorusConfig& chorus_config, Logger logger) = 0;

    /// Whether the engine can take work now.
    /// An engine that came up and later failed answers false, the same as one that never initialized.
    virtual bool is_initialized() const = 0;

    /// What this engine can do. Pre-init an envelope, post-init the effective narrowing.
    virtual EngineCapabilities capabilities() const = 0;

    /// Facts about the model in memory, or nothing until an initialize succeeds.
    virtual std::optional<LoadedModelInfo> loaded_model_info() const = 0;

    /*
     * Renders `messages` into the prompt this provider would feed the model.
     *
     * Capability flag: `EngineCapabilities::prompt_rendering`. An absent render
     * is an honest "I do not do that", never a failure, and carries no error.
     *
     * Returns:
     *  - `RenderedPrompt`: the templated text and its token count, which the
     *    runtime's fitting loop budgets against.
     *  - `std::nullopt`: the provider renders remotely or not at all, or the
     *    engine is not ready to answer.
     */
    virtual std::optional<RenderedPrompt> render_chat_prompt(
        const std::vector<ChatMessage>& messages, const std::string& template_override, bool enable_thinking
    ) const {
        (void)messages;
        (void)template_override;
        (void)enable_thinking;
        return std::nullopt;
    }

    /*
     * Answers whether this engine would accept `request`, without starting it.
     *
     * This is where a provider is honest up front: an option it does not
     * understand, a modality it cannot read, a constraint format it cannot
     * compile. Passing is not a promise the work will succeed, only that
     * nothing about the request was refusable before doing it. Errors are
     * carried in the returned rejection, never signalled.
     *
     * Returns:
     *  - `std::nullopt`: the request is acceptable.
     *  - `RequestRejection`: the code and message for the refusal.
     *
     * Errors:
     *  - `ChorusError::EngineNotReady`: the engine cannot serve work.
     *  - `ChorusError::UnsupportedFeature`: a modality or capability the provider lacks.
     *  - `ChorusError::UnsupportedOption`: an option the provider does not understand.
     */
    virtual std::optional<RequestRejection> validate_request(const ChorusRequest& request) const = 0;

    /*
     * Starts work on a `ChorusRequest` and returns before it finishes.
     *
     * Only one terminal signal, `EventType::Stop` or `EventType::Error`,
     * reaches the `ChorusRequest::on_event`. Signals arrive from an engine thread
     * or inline from this call, so a caller must be ready to see the terminal
     * before submit returns. The work outlives the call, so errors are
     * signalled on `ChorusRequest::on_event` and not returned.
     *
     * Errors:
     *  - `ChorusError::EngineNotReady`: the engine is in no state to serve.
     *  - `ChorusError::Decode`: the model failed a decode step mid-generation.
     *  - `ChorusError::Tokenize`: the prompt or a stop marker failed to tokenize.
     */
    virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;

    /*
     * Asks the engine to end request with `id` early; best-effort, returns at once.
     *
     * The request ends with a terminal signal. Signals in flight may
     * arrive after this returns. Repeated cancels of one `id` are fine.
     * Errors are signalled by `ChorusRequest::on_event` and are not returned.
     *
     * Errors:
     *  - `ChorusError::Cancelled`: the cancel reached the request before it finished.
     */
    virtual void cancel_request(RequestId id) = 0;

    /*
     * Tears the engine down.
     *
     * Work the engine holds, queued or running, ends before this returns, one
     * terminal each. Once this returns the engine invokes no
     * `ChorusRequest::on_event` it was handed, and no such invocation is in
     * progress, which lets the caller destroy whatever those callbacks write
     * into. Shutting down an idle or already shut-down engine changes nothing.
     * Errors are signalled on `ChorusRequest::on_event` and are not returned.
     *
     * Errors:
     *  - `ChorusError::Cancelled`: the engine still held the request, queued or
     *    running, when the teardown began.
     */
    virtual void shutdown() = 0;
};

} // namespace Chorus