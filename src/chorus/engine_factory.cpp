#include "chorus/engine_factory.hpp"

#include "chorus/providers/echo/echo_engine.hpp"
#include "chorus/providers/llama/llama_engine.hpp"

#include <utility>

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

InitialModelSpec make_initial_model_spec(Provider provider, std::string model_id, std::string model_source) {
    switch (provider) {
    case Provider::Echo:
        return {};
    case Provider::Llama:
        InitialModelSpec spec;
        spec.model_id = std::move(model_id);
        spec.format = ModelFormat::Gguf;
        spec.assets.push_back({AssetRole::Weights, std::move(model_source)});
        return spec;
    }
    return {};
}

EngineCapabilities describe_provider_capabilities(Provider provider) {
    // Construction is cheap and side-effect free before initialize(); the
    // contract already defines capabilities() pre-init as the provider envelope.
    auto engine = make_engine(provider);
    return engine ? engine->capabilities() : EngineCapabilities{};
}
} // namespace Chorus
