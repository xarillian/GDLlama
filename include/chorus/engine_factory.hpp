#pragma once

#include "chorus/core/inference_engine.hpp"

#include <memory>

namespace Chorus {
// The single place that knows about every concrete provider. Pure-core consumers that
// never link this module pay no llama.cpp dependency; anything calling make_engine does.
enum class Provider { Llama, Echo };

std::unique_ptr<InferenceEngine> make_engine(Provider provider);

// A provider's pre-init envelope: what it can do and what it can be configured
// with, without committing to an engine instance. Hosts render configuration
// surfaces from EngineCapabilities::load_options before any model is loaded.
EngineCapabilities describe_provider(Provider provider);
} // namespace Chorus
