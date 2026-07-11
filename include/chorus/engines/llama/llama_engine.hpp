#pragma once

#include "chorus/core/common.hpp"
#include "chorus/core/inference_engine.hpp"

#include <memory>
#include <optional>

class LlamaScheduler;

namespace Chorus {
class LlamaEngine : public InferenceEngine {
  public:
    LlamaEngine();
    ~LlamaEngine() override;

    std::optional<Chorus::ChorusError> initialize(const Chorus::ChorusConfig& config) override;
    void submit_request(const Chorus::ChorusRequest& chorus_request) override;
    void stop() override;
    bool is_initialized() const override;

  private:
    std::unique_ptr<LlamaScheduler> scheduler;
    bool _initialized = false;
    Chorus::LogCallback _log;
};
} // namespace Chorus