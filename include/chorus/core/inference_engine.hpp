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

    /// Brings the engine up under `chorus_config`, or reports why it could not.
    /// `ChorusConfig::log_callback` may be written from any engine thread.
    virtual std::optional<ChorusError> initialize(const Chorus::ChorusConfig& chorus_config) = 0;

    /// Whether the engine can take work now.
    /// An engine that came up and later failed answers false, the same as one that never initialized.
    virtual bool is_initialized() const = 0;

    virtual EngineCapabilities capabilities() const = 0;

    virtual std::optional<LoadedModelInfo> loaded_model_info() const = 0;

    // The exact templated prompt this provider would feed the model for
    // `messages`, plus its token count (the runtime's fitting loop budgets
    // against it). Providers that render remotely or not at all return
    // std::nullopt honestly; capability flag: EngineCapabilities::prompt_rendering.
    virtual std::optional<RenderedPrompt> render_chat_prompt(
        const std::vector<ChatMessage>& messages, const std::string& template_override, bool enable_thinking
    ) const {
        (void)messages;
        (void)template_override;
        (void)enable_thinking;
        return std::nullopt;
    }

    virtual std::optional<RequestRejection> validate_request(const ChorusRequest& request) const = 0;

    /*
     * Starts work on a `ChorusRequest` and returns before it finishes.
     *
     * Exactly one terminal signal, Stop or Error, reaches the request's
     * `on_event`. An engine in no state to serve refuses the work there with
     * an `EngineNotReady` error. Signals arrive from an engine thread or
     * inline from this call, so a caller must be ready to see the terminal
     * before submit returns.
     */
    virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;

    /*
     * Asks the engine to end request with `id` early; best-effort, returns at once.
     *
     * The request ends on its one terminal signal, carrying
     * `ChorusError::Cancelled` when the cancel arrives before the work
     * finishes. Signals in flight may arrive after this returns. Repeated
     * cancels of one id cost one terminal, and an id the engine does not hold
     * is inert.
     */
    virtual void cancel_request(RequestId id) = 0;

    /*
     * Tears the engine down.
     *
     * Work the engine holds, queued or running, ends before this returns: one
     * terminal each, `ChorusError::Cancelled`. Once this returns the engine
     * invokes no `on_event` it was handed, and no such invocation is in
     * progress, which lets the caller destroy whatever those callbacks write
     * into. Shutting down an idle or already shut-down engine changes nothing.
     */
    virtual void shutdown() = 0;
};

} // namespace Chorus