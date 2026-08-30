#pragma once

#include "chorus/core/inference_engine.hpp"

#include <memory>

namespace Chorus {

enum class Provider { Llama, Echo };

std::unique_ptr<InferenceEngine> make_engine(Provider provider);

/// Returns what a provider supports and how it may be configured before initialization.
EngineCapabilities describe_provider_capabilities(Provider provider);

} // namespace Chorus
