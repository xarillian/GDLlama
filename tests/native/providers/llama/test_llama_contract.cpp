#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_engine.hpp"
#include "engine_contract_suite.hpp"

#include <memory>

// Llama's binding to the shared port contract. Everything asserted here is
// asserted identically against Echo; what differs is only the shape of a
// request, which the contract has no business knowing.

namespace {

const std::string CONTRACT_MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

EngineUnderTest llama_under_test() {
    EngineUnderTest subject;
    subject.label = "Llama";
    subject.model_gated = true;
    subject.make_engine = [] { return std::make_unique<Chorus::LlamaEngine>(); };
    subject.make_config = [] {
        Chorus::ChorusConfig config;
        config.model.model_id = "test-model";
        config.model.format = Chorus::ModelFormat::Gguf;
        config.model.assets.push_back({Chorus::AssetRole::Weights, CONTRACT_MODEL_PATH});
        // One concurrent request lets the contract cases reason about work the
        // engine still holds instead of admitting the trailing request.
        config.provider_options["llama"] = Chorus::ProviderOptionMap{
            {"use_gpu", false},
            {"max_concurrent_requests", int64_t{1}},
            {"context_size", int64_t{1024}},
        };
        return config;
    };
    subject.shape_long_request = [](Chorus::ChorusRequest& request) {
        request.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
        request.gen_config.max_tokens = 512;
        request.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    };
    subject.shape_short_request = [](Chorus::ChorusRequest& request) {
        request.prompt = "hi";
        request.gen_config.max_tokens = 2;
    };
    return subject;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Llama,
    EngineContractTest,
    ::testing::Values(llama_under_test()),
    [](const ::testing::TestParamInfo<EngineUnderTest>& info) { return info.param.label; }
);
