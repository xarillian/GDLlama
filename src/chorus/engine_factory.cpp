#include "chorus/engine_factory.hpp"

#include "chorus/engines/echo/echo_engine.hpp"
#include "chorus/engines/llama/llama_engine.hpp"

namespace Chorus {
std::unique_ptr<InferenceEngine> make_engine(Backend backend) {
    switch (backend) {
    case Backend::Llama:
        return std::make_unique<LlamaEngine>();
    case Backend::Echo:
        return std::make_unique<EchoEngine>();
    }
    return nullptr; // unreachable for valid enum values; silences -Wreturn-type
}
} // namespace Chorus
