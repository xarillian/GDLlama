#include "chorus/providers/llama/llama_generation_options.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"

#include "test_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace {

std::variant<Chorus::LlamaLoadConfig, Chorus::RequestRejection> parse(Chorus::ProviderOptionMap options) {
    Chorus::ChorusConfig config;
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, "model.gguf"});
    config.provider_options["llama"] = std::move(options);
    return Chorus::parse_llama_load_config(config);
}

// One option set to `value`, every other at its declared default -- the shape a
// host produces when the user touches a single control.
std::variant<Chorus::LlamaLoadConfig, Chorus::RequestRejection>
parse_with(const std::string& key, Chorus::ProviderOptionValue value) {
    const auto& descriptors = Chorus::llama_load_option_descriptors();
    Chorus::ProviderOptionMap stored{{key, std::move(value)}};
    return parse(Chorus::resolve_option_defaults(descriptors, stored));
}

// The declared defaults must be exactly what the provider falls back to when a
// host sends nothing. This is the check that keeps a schema entry from drifting
// away from the LlamaLoadConfig member it advertises.
void test_llama_descriptor_defaults_match_the_parsed_defaults() {
    const auto parsed = parse(Chorus::resolve_option_defaults(Chorus::llama_load_option_descriptors(), {}));
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(parsed));
    const auto& got = std::get<Chorus::LlamaLoadConfig>(parsed);
    const Chorus::LlamaLoadConfig want{};

    ASSERT_EQ(got.context_size, want.context_size);
    ASSERT_EQ(got.thread_count, want.thread_count);
    ASSERT_TRUE(got.use_gpu == want.use_gpu);
    ASSERT_EQ(got.gpu_layers, want.gpu_layers);
    ASSERT_EQ(got.num_slots, want.num_slots);
    ASSERT_EQ(got.tokens_per_tick, want.tokens_per_tick);
    ASSERT_EQ(got.n_batch, want.n_batch);
    ASSERT_EQ(got.n_ubatch, want.n_ubatch);
    ASSERT_EQ(got.main_gpu, want.main_gpu);
}

// The contract's half of a widget bound: whatever a host lets a user pick
// inside the declared range, the provider accepts. A hint tightened past the
// provider's own limits, or a limit tightened past the hint, fails here rather
// than in a game at load time.
void test_llama_descriptor_bounds_are_accepted() {
    for (const auto& descriptor : Chorus::llama_load_option_descriptors()) {
        if (!descriptor.minimum || !descriptor.maximum)
            continue;
        for (const double bound : {*descriptor.minimum, *descriptor.maximum}) {
            const auto parsed = parse_with(descriptor.key, int64_t(bound));
            if (!std::holds_alternative<Chorus::LlamaLoadConfig>(parsed)) {
                std::cout << "    rejected " << descriptor.key << " = " << bound << ": "
                          << std::get<Chorus::RequestRejection>(parsed).message << "\n";
                ASSERT_TRUE(false);
            }
        }
    }
}

void test_llama_rejects_options_outside_the_declaration() {
    const auto unknown = parse({{"not_a_real_option", int64_t{1}}});
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(unknown));
    ASSERT_TRUE(std::get<Chorus::RequestRejection>(unknown).error == Chorus::ChorusError::UnsupportedOption);

    const auto mistyped = parse({{"context_size", std::string("large")}});
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(mistyped));
    ASSERT_TRUE(std::get<Chorus::RequestRejection>(mistyped).error == Chorus::ChorusError::UnsupportedOption);
}

void test_llama_rejects_nonpositive_runtime_dimensions() {
    for (const auto& [key, value] : std::vector<std::pair<std::string, int64_t>>{
             {"context_size", 0},
             {"thread_count", 0},
             {"gpu_layers", -2},
             {"num_slots", 0},
             {"tokens_per_tick", 0},
         }) {
        const auto parsed = parse({{key, value}});
        const auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed);
        ASSERT_TRUE(rejection != nullptr);
        ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
        ASSERT_TRUE(rejection->message.find(key) != std::string::npos);
    }
}

void test_llama_rejects_runtime_dimensions_that_do_not_fit() {
    const int64_t above_int32 = int64_t{std::numeric_limits<int32_t>::max()} + 1;
    const int64_t above_uint32 = int64_t{std::numeric_limits<uint32_t>::max()} + 1;
    for (const auto& [key, value] : std::vector<std::pair<std::string, int64_t>>{
             {"context_size", above_uint32},
             {"thread_count", above_int32},
             {"gpu_layers", above_int32},
             {"num_slots", above_uint32},
             {"tokens_per_tick", above_int32},
         }) {
        const auto parsed = parse({{key, value}});
        const auto* rejection = std::get_if<Chorus::RequestRejection>(&parsed);
        ASSERT_TRUE(rejection != nullptr);
        ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
        ASSERT_TRUE(rejection->message.find(key) != std::string::npos);
    }
}

// The declaration is advice to hosts, not enforcement: a host that ignores it
// and sends an option whose prerequisite is off still gets an honest rejection.
void test_llama_options_with_unmet_prerequisite_still_reject_when_sent() {
    const auto parsed = parse({{"use_gpu", false}, {"main_gpu", int64_t{2}}});
    ASSERT_TRUE(std::holds_alternative<Chorus::RequestRejection>(parsed));

    // ... and the declared prerequisite is what keeps a host from sending them.
    const auto resolved =
        Chorus::resolve_option_defaults(Chorus::llama_load_option_descriptors(), {{"use_gpu", false}});
    ASSERT_TRUE(resolved.find("main_gpu") == resolved.end());
    ASSERT_TRUE(resolved.find("gpu_layers") == resolved.end());
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(parse(resolved)));
}

// The Godot request normalizer promotes 'repeat_penalty' to a top-level
// convenience key over provider_options["llama"]. That spelling only works
// while llama still declares the option it forwards to.
void test_normalizer_convenience_keys_exist_in_the_provider_declaration() {
    const auto& names = Chorus::llama_provider_generation_option_names();
    ASSERT_TRUE(std::find(names.begin(), names.end(), "repeat_penalty") != names.end());
}

} // namespace

int run_llama_load_option_tests() {
    std::cout << "\n--- Llama Load Option Schema Tests ---\n";
    run_test(
        "Llama descriptor defaults match the parsed defaults", test_llama_descriptor_defaults_match_the_parsed_defaults
    );
    run_test("Llama descriptor bounds are accepted", test_llama_descriptor_bounds_are_accepted);
    run_test("Llama rejects options outside the declaration", test_llama_rejects_options_outside_the_declaration);
    run_test("Llama rejects nonpositive runtime dimensions", test_llama_rejects_nonpositive_runtime_dimensions);
    run_test(
        "Llama rejects runtime dimensions that do not fit", test_llama_rejects_runtime_dimensions_that_do_not_fit
    );
    run_test(
        "Llama options with unmet prerequisite still reject when sent",
        test_llama_options_with_unmet_prerequisite_still_reject_when_sent
    );
    run_test(
        "Normalizer convenience keys exist in the provider declaration",
        test_normalizer_convenience_keys_exist_in_the_provider_declaration
    );
    return 0;
}
