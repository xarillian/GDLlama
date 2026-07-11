#include "test_utils.hpp"
#include "chorus/engines/llama/llama_utils.hpp"

#include "llama.h"

#include <iostream>
#include <string>

static const std::string UTILS_MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

// RAII helper that loads a minimal llama model + context for use in tests.
struct LlamaContextFixture {
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

void test_tokenize_resizes_and_retries_on_overflow() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    LlamaContextFixture fixture;
    ASSERT_TRUE(fixture.load());

    std::string short_prompt = "Hi!";
    auto tokens = Chorus::LlamaUtils::tokenize(fixture.context, short_prompt, /*add_special=*/true);

    ASSERT_TRUE(!tokens.empty());
    ASSERT_TRUE(static_cast<int>(tokens.size()) <= static_cast<int>(short_prompt.length()) + 4);
}

void test_tokenize_returns_tokens_for_normal_ascii_input() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    LlamaContextFixture fixture;
    ASSERT_TRUE(fixture.load());

    std::string normal_prompt = "Hello, world!";
    auto tokens = Chorus::LlamaUtils::tokenize(fixture.context, normal_prompt, /*add_special=*/true);

    ASSERT_TRUE(!tokens.empty());
}

void test_tokenize_returns_special_tokens_for_empty_string_with_special_tokens_enabled() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    LlamaContextFixture fixture;
    ASSERT_TRUE(fixture.load());

    std::string empty_prompt = "";
    auto tokens = Chorus::LlamaUtils::tokenize(fixture.context, empty_prompt, /*add_special=*/true);

    ASSERT_TRUE(!tokens.empty());
}

void test_tokenize_returns_empty_for_empty_string_with_special_tokens_disabled() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    LlamaContextFixture fixture;
    ASSERT_TRUE(fixture.load());

    std::string empty_prompt = "";
    auto tokens = Chorus::LlamaUtils::tokenize(fixture.context, empty_prompt, /*add_special=*/false);

    ASSERT_TRUE(tokens.empty());
}

// ---------------------------------------------------------------------------
// Suite entry point
// ---------------------------------------------------------------------------

int run_llama_utils_tests() {
    std::cout << "\n--- LLAMA UTILS SUITE ---\n";

    run_test("Tokenize_ResizesAndRetriesOnOverflow", test_tokenize_resizes_and_retries_on_overflow);
    run_test("Tokenize_ReturnsTokensForNormalAsciiInput", test_tokenize_returns_tokens_for_normal_ascii_input);
    run_test(
        "Tokenize_ReturnsSpecialTokensForEmptyStringWithSpecialTokensEnabled",
        test_tokenize_returns_special_tokens_for_empty_string_with_special_tokens_enabled
    );
    run_test(
        "Tokenize_ReturnsEmptyForEmptyStringWithSpecialTokensDisabled",
        test_tokenize_returns_empty_for_empty_string_with_special_tokens_disabled
    );

    return g_tests_failed > 0 ? 1 : 0;
}
