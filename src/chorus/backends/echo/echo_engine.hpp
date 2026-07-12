#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace Chorus {
// Reference implementation of InferenceEngine: model-free and dependency-free.
//
// Echoes each request's prompt back word-by-word as Token signals (their concatenation
// equals the prompt exactly), then a Stop, from a worker thread mirroring the real
// engines' async contract. Exists to prove the backend seam, to let CI and day-one
// integrations exercise the full signal path with no model file, and to keep the
// interface contract compiler-enforced as it grows. It is not a gameplay test space:
// echoed prompts say nothing about how real dialogue reads or paces.
class EchoEngine : public InferenceEngine {
  public:
    EchoEngine();
    ~EchoEngine() override;

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config) override;
    void submit_request(const Chorus::ChorusRequest& chorus_request) override;
    void stop() override;
    bool is_initialized() const override;

  private:
    void worker_loop();

    std::queue<Chorus::ChorusRequest> _queue;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    std::thread _worker;
    std::atomic<bool> _running{false};
    bool _initialized = false;
    Chorus::LogCallback _log;
};
} // namespace Chorus
