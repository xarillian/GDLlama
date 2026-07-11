#pragma once

#include "chorus/core/common.hpp"

#include <optional>

namespace Chorus {

class InferenceEngine {
  public:
    virtual ~InferenceEngine() = default;

    virtual std::optional<ChorusError> initialize(const Chorus::ChorusConfig& config) = 0;
    virtual bool is_initialized() const = 0;

    virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;

    virtual void stop() = 0;
};

} // namespace Chorus