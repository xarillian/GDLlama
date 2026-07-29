#pragma once

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"

#include <optional>

namespace Chorus {

// The provider port. Implementations must honor the callback contract:
//
// - ChorusRequest::on_event may be invoked from an engine worker thread, or
//   inline on the caller's thread during submit_request (e.g. the
//   EngineNotReady rejection path). Callbacks must therefore be thread-safe.
// - After stop() returns, the engine must never invoke a previously supplied
//   on_event again. (Current engines guarantee this by joining their worker
//   inside stop().) ChorusRuntime relies on this to destroy its event queue
//   safely.
// - ChorusConfig::log_callback may be invoked from any engine thread; hosts
//   must supply a thread-safe sink.
class InferenceEngine {
  public:
    virtual ~InferenceEngine() = default;

    virtual std::optional<ChorusError> initialize(const Chorus::ChorusConfig& config) = 0;
    virtual bool is_initialized() const = 0;

    virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;
    virtual void cancel_request(RequestId id) = 0;

    virtual void stop() = 0;

    // --- Capability self-description (all three are host-thread-only, like
    // initialize/stop: they read state those methods mutate; they are never
    // called from engine workers) ---

    // Pre-init: the provider envelope. Post-init: the effective intersection
    // of provider, model, and load configuration.
    virtual EngineCapabilities capabilities() const = 0;

    virtual std::optional<LoadedModelInfo> loaded_model_info() const = 0;

    // Lightweight, side-effect-free synchronous check: readiness, request
    // type, constraint format, and named options. Never rejects a request
    // for carrying a session id. Acceptance does not guarantee execution
    // cannot fail later.
    virtual std::optional<RequestRejection> validate_request(const ChorusRequest& request) const = 0;

    // --- Optional prompt rendering (host-thread-only, like capabilities) ---

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
};

} // namespace Chorus