#include <gtest/gtest.h>
#include "llama_controller.hpp"

class LlamaControllerTest : public ::testing::Test {
    protected:
        std::unique_ptr<LlamaController> controller;
        common_params params;

        void SetUp() override {
            controller = std::make_unique<LlamaController>();
            params.model.path = "tests/models/gemma-3-270m-it-F16.gguf";
            params.n_ctx = 256;
            params.n_gpu_layers = 0;
        }

        void TearDown() override {
            if (controller && controller->is_model_loaded()) {
                controller->unload_model();
            }
        }
};

TEST_F(LlamaControllerTest, GenerateThrowsErrorWhenModelNotLoaded) {
    ASSERT_FALSE(controller->is_model_loaded());

    ASSERT_THROW(controller->start_generation(
        params, "test prompt", "", "", false, nullptr
    ), std::runtime_error);
}

TEST_F(LlamaControllerTest, ResetContextClearsConversationHistory) {
    controller->load_model(params);
    ASSERT_TRUE(controller->is_model_loaded());
    
    std::string response1 = controller->start_generation(
        params, "Hello!", "", "", true, // is_continuous = true
        [](const std::string& chunk) {}
    );
    ASSERT_EQ(controller->get_conversation_history().size(), 2);
    
    controller->reset_context();

    ASSERT_TRUE(controller->get_conversation_history().empty());
}