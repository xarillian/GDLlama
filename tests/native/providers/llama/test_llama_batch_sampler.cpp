#include "chorus/providers/llama/llama_batch_sampler.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_generation_options.hpp"
#include "silent_llama_log.hpp"
#include "gtest_utils.hpp"

class LlamaBatchSamplerModelTest : public ChorusModelTest {};

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

llama_token zero_temperature_greedy_chain(std::vector<float> logits) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    std::vector<llama_token_data> candidates;
    for (size_t token = 0; token < logits.size(); ++token)
        candidates.push_back({static_cast<llama_token>(token), logits[token], 0.0f});
    llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
    llama_sampler_apply(chain, &array);
    llama_sampler_free(chain);
    return array.data[array.selected].id;
}

std::vector<float> row_of(float fill, std::initializer_list<std::pair<size_t, float>> values) {
    std::vector<float> row(1003, fill);
    for (const auto& [index, value] : values)
        row[index] = value;
    return row;
}

Chorus::LlamaSamplingPath path_for(llama_model* model, float temperature, Chorus::ProviderOptionMap options) {
    Chorus::GenerationConfig config;
    config.temperature = temperature;
    config.provider_options["llama"] = std::move(options);
    auto resolved = std::get<Chorus::ResolvedLlamaGeneration>(Chorus::resolve_llama_generation(config));
    auto sampler = std::get<common_sampler_ptr>(Chorus::make_llama_sampler(model, resolved));
    return Chorus::llama_sampler_selects_first_maximum(sampler.get()) ? Chorus::LlamaSamplingPath::FirstMaximum
                                                                      : Chorus::LlamaSamplingPath::Chain;
}

} // namespace

TEST(LlamaBatchSampler, First_maximum_picks_what_a_zero_temperature_greedy_chain_picks) {
    std::mt19937 random(7);
    std::normal_distribution<float> logit(0.0f, 4.0f);
    std::vector<float> noisy(1003);
    for (float& value : noisy)
        value = logit(random);

    const std::vector<std::vector<float>> rows{
        noisy,
        row_of(0.0f, {{600, 9.0f}, {300, 9.0f}}),         // tie across lane chunks
        row_of(0.0f, {{0, kNan}, {10, 9.0f}}),            // NaN where the scan starts
        row_of(0.0f, {{251, kNan}, {260, 9.0f}}),         // NaN at a lane chunk's start
        row_of(-kInfinity, {}),                           // no finite logit
        row_of(-kInfinity, {{900, -1.0f}}),               // one finite logit late in the row
        row_of(1.0f, {{2, kInfinity}, {700, kInfinity}}), // tied infinities
        row_of(kNan, {}),                                 // only NaN
    };
    wlib::ParallelFor parallel(4);

    for (size_t row = 0; row < rows.size(); ++row) {
        const llama_token expected = zero_temperature_greedy_chain(rows[row]);
        EXPECT_EQ(Chorus::llama_first_maximum(rows[row], nullptr), expected) << "row " << row;
        EXPECT_EQ(Chorus::llama_first_maximum(rows[row], &parallel), expected) << "row " << row << " across lanes";
    }
}

TEST_F(LlamaBatchSamplerModelTest, Only_a_temperature_only_zero_temperature_chain_takes_the_first_maximum) {
    SilentLlamaLog quiet;
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file("tests/models/gemma-3-270m-it-F16.gguf", params);
    ASSERT_NE(model, nullptr);
    const Chorus::ProviderOptionMap temperature_only{{"sampler_order", Chorus::ProviderOptionList{std::string("temperature")}}};

    EXPECT_EQ(path_for(model, 0.0f, temperature_only), Chorus::LlamaSamplingPath::FirstMaximum);
    EXPECT_EQ(path_for(model, 0.7f, temperature_only), Chorus::LlamaSamplingPath::Chain);
    EXPECT_EQ(path_for(model, 0.0f, {}), Chorus::LlamaSamplingPath::Chain);
    EXPECT_EQ(path_for(model, 0.0f, {{"sampler_order", Chorus::ProviderOptionList{std::string("temperature")}},
                                     {"logit_bias", Chorus::ProviderOptionMap{{"5", 1.0}}}}),
              Chorus::LlamaSamplingPath::Chain);
    llama_model_free(model);
}
