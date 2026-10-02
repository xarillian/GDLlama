#pragma once

#include "chorus/core/inference_engine.hpp"

#include <memory>
#include <string>

namespace Chorus {

enum class Provider { Llama, Echo };

std::unique_ptr<InferenceEngine> make_engine(Provider provider);

/// Shapes a host-selected model according to the provider's asset policy.
InitialModelSpec make_initial_model_spec(Provider provider, std::string model_id, std::string model_source);

/// Returns what a provider supports and how it may be configured before initialization.
EngineCapabilities describe_provider_capabilities(Provider provider);

} // namespace Chorus
