#pragma once

#include "chorus/core/common.hpp"

#include <optional>

namespace Chorus {

// The backend port. Implementations must honor the callback contract:
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

    virtual void stop() = 0;
};

} // namespace Chorus