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

    /// Brings the engine up under `config`, or reports why it could not.
    /// `ChorusConfig::log_callback` may be written from any engine thread.
    virtual std::optional<ChorusError> initialize(const Chorus::ChorusConfig& config) = 0;

    /// Whether the engine can take work now. An engine that came up and later
    /// failed answers false, the same as one that never initialized.
    virtual bool is_initialized() const = 0;

    // Pre-init: the provider envelope. Post-init: the effective intersection
    // of provider, model, and load configuration.
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

    // Lightweight, side-effect-free synchronous check: readiness, request
    // type, constraint format, and named options. Never rejects a request
    // for carrying a session id. Acceptance does not guarantee execution
    // cannot fail later.
    virtual std::optional<RequestRejection> validate_request(const ChorusRequest& request) const = 0;

    /// Takes the request on. `on_event` fires from an engine worker, or inline
    /// before this returns when the engine rejects the request outright.
    virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;

    virtual void cancel_request(RequestId id) = 0;

    /*
     * Tears the engine down, fencing its callbacks.
     *
     * Once this returns, an `on_event` the engine was given before is never
     * invoked again, which is what lets a caller then destroy its sinks.
     */
    virtual void stop() = 0;
};

} // namespace Chorus