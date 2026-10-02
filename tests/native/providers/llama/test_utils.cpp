#include "gtest_utils.hpp"

class LlamaUtilsModelTest : public ChorusModelTest {};
#include "chorus/providers/llama/llama_utils.hpp"
#include "silent_llama_log.hpp"

#include "llama.h"

#include <iostream>
#include <string>

static const std::string UTILS_MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

// RAII helper that loads a minimal llama model + context for use in tests.
struct LlamaContextFixture {
    // First member, so it outlives the frees below and llama stays quiet
    // through teardown as well as load.
    SilentLlamaLog quiet;
    llama_model* model = nullptr;
    llama_context* context = nullptr;

    bool load() {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model = llama_model_load_from_file(UTILS_MODEL_PATH.c_str(), model_params);
        if (!model)
            return false;

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 512;
        context = llama_init_from_model(model, context_params);
        return context != nullptr;
    }

    ~LlamaContextFixture() {
        if (context)
            llama_free(context);
        if (model)
            llama_model_free(model);
    }
};

// ---------------------------------------------------------------------------
// tokenize() Tests
// ---------------------------------------------------------------------------

TEST_F(LlamaUtilsModelTest, Tokenize_empty_string_distinguishes_special_token_modes) {

    LlamaContextFixture fixture;
    ASSERT_TRUE(fixture.load());

    const std::string empty_prompt;
    const auto with_special = Chorus::LlamaUtils::tokenize(fixture.context, empty_prompt, /*add_special=*/true);
    const auto without_special = Chorus::LlamaUtils::tokenize(fixture.context, empty_prompt, /*add_special=*/false);

    ASSERT_TRUE(!with_special.empty());
    ASSERT_TRUE(without_special.empty());
}
