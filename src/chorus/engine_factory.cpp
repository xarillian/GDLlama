#include "chorus/engine_factory.hpp"

#include "chorus/providers/echo/echo_engine.hpp"
#include "chorus/providers/llama/llama_engine.hpp"

namespace Chorus {
std::unique_ptr<InferenceEngine> make_engine(Provider provider) {
    switch (provider) {
    case Provider::Llama:
        return std::make_unique<LlamaEngine>();
    case Provider::Echo:
        return std::make_unique<EchoEngine>();
    }
    return nullptr; // unreachable for valid enum values; silences -Wreturn-type
}

EngineCapabilities describe_provider(Provider provider) {
    // Construction is cheap and side-effect free before initialize(); the
    // contract already defines capabilities() pre-init as the provider envelope.
    auto engine = make_engine(provider);
    return engine ? engine->capabilities() : EngineCapabilities{};
}
} // namespace Chorus
