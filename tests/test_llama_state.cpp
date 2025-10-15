#include <gtest/gtest.h>
#include "llama_state.hpp"
#include "common.h"

class LlamaStateTest : public ::testing::Test {
    protected:
        LlamaState llama_state;
        common_params params;

        void SetUp() override {
            params.model.path = "tests/models/gemma-3-270m-it-F16.gguf";
            params.n_ctx = 128;
            params.n_gpu_layers = 0;  // Use CPU for testing
        }
};

TEST_F(LlamaStateTest, CanLoadModel) {
    bool success = llama_state.load(params);
    ASSERT_TRUE(success);
    ASSERT_TRUE(llama_state.is_loaded());
    ASSERT_NE(llama_state.get_model(), nullptr);
    ASSERT_NE(llama_state.get_context(), nullptr);

    llama_state.unload();
}

TEST_F(LlamaStateTest, CanUnloadModel) {
    llama_state.load(params);

    llama_state.unload();
    ASSERT_FALSE(llama_state.is_loaded());
    ASSERT_EQ(llama_state.get_model(), nullptr);
    ASSERT_EQ(llama_state.get_context(), nullptr);
}

TEST_F(LlamaStateTest, UnloadingWhenNotLoadedDoesNothing) {
    ASSERT_FALSE(llama_state.is_loaded());
    llama_state.unload();  // no crash pl0x
    ASSERT_FALSE(llama_state.is_loaded());
}
