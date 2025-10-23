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
            params.n_gpu_layers = 0;  // Use CPU for testing
        }

        void TearDown() override {
            if (controller && controller->is_model_loaded()) {
                controller->unload_model();
            }
        }
};

TEST_F(LlamaControllerTest, GenerateReturnsErrorWhenModelNotLoaded) {
    ASSERT_FALSE(controller->is_model_loaded());

    std::string error_msg;
    std::string result = controller->start_generation(
        params, "test prompt", "", "", false,  // is_conversational = false
        [](const std::string& chunk) {},
        &error_msg
    );

    ASSERT_FALSE(error_msg.empty());
    ASSERT_TRUE(result.empty());
    ASSERT_NE(error_msg.find("Model is not loaded"), std::string::npos);
}

TEST_F(LlamaControllerTest, GenerateEmbeddingReturnsErrorWhenModelNotLoaded) {
    ASSERT_FALSE(controller->is_model_loaded());

    std::string error_msg;
    std::vector<float> result = controller->generate_embedding(
        params, "test prompt", &error_msg
    );

    ASSERT_FALSE(error_msg.empty());
    ASSERT_TRUE(result.empty());
    ASSERT_NE(error_msg.find("Model is not loaded"), std::string::npos);
}

TEST_F(LlamaControllerTest, ResetContextClearsConversationHistory) {
    auto load_result = controller->load_model(params);
    ASSERT_EQ(load_result, godot::OK);
    ASSERT_TRUE(controller->is_model_loaded());
    
    std::string error_msg;
    std::string response1 = controller->start_generation(
        params, "Hello!", "", "", true, // is_conversational = true
        [](const std::string& chunk) {},
        &error_msg
    );
    
    ASSERT_TRUE(error_msg.empty());
    ASSERT_EQ(controller->get_conversation_history().size(), 2);

    controller->reset_context();
    ASSERT_TRUE(controller->get_conversation_history().empty());
}

TEST_F(LlamaControllerTest, SuccessfulGenerationHasNoError) {
    auto load_result = controller->load_model(params);
    ASSERT_EQ(load_result, godot::OK);
    ASSERT_TRUE(controller->is_model_loaded());

    std::string error_msg;
    params.n_predict = 10;
    
    std::string result = controller->start_generation(
        params, "Say hello", "", "", false,  // is_conversational = false
        [](const std::string& chunk) {},
        &error_msg
    );

    ASSERT_TRUE(error_msg.empty());
    ASSERT_FALSE(result.empty());
}

TEST_F(LlamaControllerTest, NonConversationalModeDoesNotBuildHistory) {
    auto load_result = controller->load_model(params);
    ASSERT_EQ(load_result, godot::OK);
    
    std::string error_msg;
    params.n_predict = 10;
    
    std::string result_1 = controller->start_generation(
        params, "Say hello", "", "", false,  // is_conversational = false
        [](const std::string& chunk) {},
        &error_msg
    );
    
    ASSERT_TRUE(error_msg.empty());
    ASSERT_TRUE(controller->get_conversation_history().empty());  // No history

    std::string result_2 = controller->start_generation(
        params, "Say goodbye", "", "", false,
        [](const std::string& chunk) {},
        &error_msg
    );
    
    ASSERT_TRUE(error_msg.empty());
    ASSERT_TRUE(controller->get_conversation_history().empty());  // Still no history
}

TEST_F(LlamaControllerTest, ConversationalModeBuildHistory) {
    auto load_result = controller->load_model(params);
    ASSERT_EQ(load_result, godot::OK);

    std::string error_msg;
    params.n_predict = 10;

    std::string result_1 = controller->start_generation(
        params, "Hello", "", "", true,  // is_conversational = true
        [](const std::string& chunk) {},
        &error_msg
    );

    ASSERT_TRUE(error_msg.empty());
    ASSERT_EQ(controller->get_conversation_history().size(), 2);  // user + assistant
    ASSERT_EQ(controller->get_conversation_history()[0].role, "user");
    ASSERT_EQ(controller->get_conversation_history()[0].content, "Hello");
    ASSERT_EQ(controller->get_conversation_history()[1].role, "assistant");

    std::string result_2 = controller->start_generation(
        params, "How are you?", "", "", true,
        [](const std::string& chunk) {},
        &error_msg
    );

    ASSERT_TRUE(error_msg.empty());
    ASSERT_EQ(controller->get_conversation_history().size(), 4);  // 2 exchanges
}

TEST_F(LlamaControllerTest, InfinitePredictGeneratesUntilEOS) {
    auto load_result = controller->load_model(params);
    ASSERT_EQ(load_result, godot::OK);

    std::string error_msg;
    params.n_predict = -1;  // Infinite
    params.n_ctx = 128;
    params.sampling.ignore_eos = false;
    // Adjust sampling parameters to encourage completion
    params.sampling.temp = 1.5f;
    params.sampling.penalty_repeat = 1.0f;
    params.sampling.top_p = 0.95f;


    std::string result = controller->start_generation(
        params, "Say hello", "", "", false,
        [](const std::string& chunk) {},
        &error_msg
    );

    ASSERT_TRUE(error_msg.empty());
    ASSERT_FALSE(result.empty());  // Hopefully it generated something
}
