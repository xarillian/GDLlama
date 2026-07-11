#pragma once

#include "chorus/core/inference_engine.hpp"

#include <memory>

namespace Chorus {
// The single place that knows about every concrete backend. Pure-core consumers that
// never link this module pay no llama.cpp dependency; anything calling make_engine does.
enum class Backend { Llama, Echo };

std::unique_ptr<InferenceEngine> make_engine(Backend backend);
} // namespace Chorus
