#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

namespace Chorus {
// Reference implementation of InferenceEngine: model-free and dependency-free.
//
// Echoes each request's prompt back word-by-word as Token signals (their concatenation
// equals the prompt exactly), then a Stop, from a worker thread mirroring the real
// engines' async contract. Exists to prove the provider seam, to let CI and day-one
// integrations exercise the full signal path with no model file, and to keep the
// interface contract compiler-enforced as it grows. It is not a gameplay test space:
// echoed prompts say nothing about how real dialogue reads or paces.
class EchoEngine : public InferenceEngine {
  public:
    EchoEngine();
    ~EchoEngine() override;

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config) override;
    void submit_request(const Chorus::ChorusRequest& chorus_request) override;
    void cancel_request(RequestId id) override;
    void stop() override;
    bool is_initialized() const override;

    EngineCapabilities capabilities() const override;
    std::optional<LoadedModelInfo> loaded_model_info() const override;
    std::optional<RequestRejection> validate_request(const Chorus::ChorusRequest& request) const override;

  private:
    void worker_loop();

    std::deque<Chorus::ChorusRequest> _queue;
    std::optional<Chorus::ChorusRequest> _active;
    std::unordered_set<RequestId> _cancelled_ids;
    size_t _queued_cancel_callbacks_in_flight = 0;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    std::thread _worker;
    bool _running = false;
    bool _initialized = false;
    Chorus::LogCallback _log;
    // Content controls are accepted-and-inert (see validate_request); the
    // one-per-lifetime warning keeps the discard from being silent.
    mutable std::atomic<bool> _warned_ignored{false};
};
} // namespace Chorus
