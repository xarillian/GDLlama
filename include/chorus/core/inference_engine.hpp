#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <optional>

namespace Chorus {

/*
 * The service contract implemented by every inference provider.
 *
 * Callers invoke engine methods from their confined thread. An engine may
 * invoke request callbacks from any of its worker threads, so every callback
 * must be thread-safe. Engine workers never invoke these methods.
 */
class InferenceEngine {
  public:
    virtual ~InferenceEngine() = default;

    /*
     * Initializes the engine from `chorus_config`.
     *
     * Calling this on an initialized engine succeeds without changing it. An
     * engine that failed after initialization releases its existing resources
     * before acquiring replacements, so two models are never resident at once.
     *
     * Initialization failures are returned instead of signalled.
     *
     * Returns:
     *  - `std::nullopt`: initialization succeeded.
     *  - `ChorusError`: initialization failed.
     *
     * Errors:
     *  - `ChorusError::UnsupportedModelFormat`: the provider cannot read the artifact.
     *  - `ChorusError::ModelLoad`: the weights failed to load.
     *  - `ChorusError::ContextInit`: the inference context failed to initialize.
     *  - `ChorusError::UnsupportedOption`: a load option the provider does not declare.
     *  - `ChorusError::Unknown`: the provider could not classify the failure.
     */
    virtual std::optional<ChorusError> initialize(const Chorus::ChorusConfig& chorus_config, Logger logger) = 0;

    /// Whether the engine is ready to accept work.
    /// An engine that fails after initialization returns false, as does an
    /// engine that has never been initialized.
    virtual bool is_initialized() const = 0;

    /// Returns the capability envelope before initialization and the effective capabilities afterward.
    virtual EngineCapabilities capabilities() const = 0;

    /// Returns facts about the model in memory, or `std::nullopt` when none is loaded.
    virtual std::optional<LoadedModelInfo> loaded_model_info() const = 0;

    /*
     * Renders `messages` into the prompt this provider would feed the model.
     *
     * Capability flag: `EngineCapabilities::prompt_rendering`. An absent value
     * means the provider does not render locally or the engine is not ready; it
     * is not itself a failure.
     *
     * Returns:
     *  - `RenderedPrompt`: the templated text and token count used by the
     *    runtime's fitting loop.
     *  - `std::nullopt`: no local render is available.
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
     * A provider rejects anything it cannot support before work begins, such
     * as an unknown option, an unsupported modality, or an unusable constraint
     * format. Passing means only that nothing in the request required a
     * preflight rejection. The returned rejection carries the error; nothing
     * is signalled.
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
     * Starts work on `chorus_request`.
     *
     * Exactly one terminal signal, `ChorusSignal::Stop` or
     * `ChorusSignal::Error`, reaches `ChorusRequest::on_event`. It may arrive
     * inline before this method returns or later from an engine thread. The
     * request may outlive this call, so failures are signalled on
     * `ChorusRequest::on_event` and not returned.
     */
    virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;

    /*
     * Requests cancellation of the request with `id` and returns immediately.
     *
     * Cancellation is best-effort. Signals already in flight may arrive after
     * this method returns, and repeated cancellation requests for one `id` are
     * safe. The request still terminates exactly once. Cancellation is
     * signalled on `ChorusRequest::on_event` and not returned.
     */
    virtual void cancel_request(RequestId id) = 0;

    /*
     * Stops all work and tears the engine down.
     *
     * Every queued or running request terminates exactly once before this
     * method returns. Afterward, no invocation of a previously supplied
     * `ChorusRequest::on_event` is running or can begin, so callers may safely
     * destroy callback state. Calling this on an idle or stopped engine changes
     * nothing. Cancellation is signalled on `ChorusRequest::on_event` and not
     * returned.
     */
    virtual void shutdown() = 0;
};

} // namespace Chorus