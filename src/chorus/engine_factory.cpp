#include "chorus/engine_factory.hpp"

#include "chorus/backends/echo/echo_engine.hpp"
#include "chorus/backends/llama/llama_engine.hpp"

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

EngineCapabilities describe_backend(Backend backend) {
    // Construction is cheap and side-effect free before initialize(); the
    // contract already defines capabilities() pre-init as the backend envelope.
    auto engine = make_engine(backend);
    return engine ? engine->capabilities() : EngineCapabilities{};
}
} // namespace Chorus
