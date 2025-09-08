#include "llama_controller.hpp"
#include "llama_runner.hpp"
#include "gtest/gtest.h"
#include <common/json-schema-to-grammar.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <atomic>
#include <chrono>

// A mock version of LlamaRunner we can use to record test progression.
class MockLlamaRunner : public LlamaRunner {
public:
    common_params last_params_received;
    std::string last_prompt_received;
    std::atomic<int> call_count{0};
    std::atomic<bool> is_processing{false};

    MockLlamaRunner(bool should_output_prompt = true)
        : LlamaRunner(should_output_prompt) {}

    std::string llama_generate_text(
        std::string prompt,
        common_params params,
        std::function<void(std::string)> on_generate_text_updated,
        std::function<void()> on_input_wait_started,
        std::function<void(std::string)> on_generate_text_finished
    ) override {
        last_prompt_received = prompt;
        last_params_received = params;
        call_count++;

        // Check for concurrent access to fail the mutex test if needed.
        bool already_processing = is_processing.exchange(true);
        EXPECT_FALSE(already_processing) << "Mutex failed: Another thread entered a critical section.";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
        is_processing.store(false);

        on_generate_text_updated("This is a ");
        on_generate_text_updated("mocked text generation.");
        on_generate_text_finished("This is a mocked text generation.");
        
        return "This is a mocked text generation.";
    }
};


TEST(LlamaControllerTest, GrammarIsPrioritizedOverJson) {
    // Setup
    LlamaController controller;
    auto mock_runner = std::make_unique<MockLlamaRunner>();
    MockLlamaRunner* mock_ptr = mock_runner.get();
    controller.set_llama_runner(std::move(mock_runner));

    std::string grammar_str = "root ::= \"a\"";
    std::string json_str = "{\"type\": \"string\"}";

    // Execute
    controller.generate_text_locked(
        "prompt", grammar_str, json_str,
        [](auto s){}, [](){}, [](auto s){}
    );

    // Assert
    EXPECT_EQ(mock_ptr->last_params_received.sampling.grammar, grammar_str);
}

TEST(LlamaControllerTest, JsonIsUsedWhenGrammarIsEmpty) {
    // Setup
    LlamaController controller;
    auto mock_runner = std::make_unique<MockLlamaRunner>();
    MockLlamaRunner* mock_ptr = mock_runner.get();
    controller.set_llama_runner(std::move(mock_runner));

    std::string json_str = "{\"type\": \"string\"}";
    std::string expected_grammar = json_schema_to_grammar(nlohmann::ordered_json::parse(json_str));

    // Execute
    controller.generate_text_locked(
        "prompt", "", json_str,
        [](auto s){}, [](){}, [](auto s){}
    );

    // Assert
    EXPECT_EQ(mock_ptr->last_params_received.sampling.grammar, expected_grammar);
}

TEST(LlamaControllerTest, CallbacksAreInvokedCorrectly) {
    // Setup
    LlamaController controller;
    controller.set_llama_runner(std::make_unique<MockLlamaRunner>());

    bool update_was_called = false;
    bool wait_start_was_called = false;
    bool finish_was_called = false;

    // Execute: Pass lambdas that modify our boolean flags.
    controller.generate_text_locked(
        "prompt", "", "",
        [&](auto s){ update_was_called = true; },
        [&](){ wait_start_was_called = true; },
        [&](auto s){ finish_was_called = true; }
    );

    // Assert
    EXPECT_TRUE(update_was_called);
    // Note: on_input_wait_started is not called in this mock's path, so it remains false.
    // EXPECT_TRUE(wait_start_was_called); 
    EXPECT_TRUE(finish_was_called);
}

TEST(LlamaControllerTest, MutexPreventsConcurrentAccess) {
    // Setup
    auto controller_shared = std::make_shared<LlamaController>();
    auto mock_runner = std::make_unique<MockLlamaRunner>();
    MockLlamaRunner* mock_ptr = mock_runner.get();
    controller_shared->set_llama_runner(std::move(mock_runner));

    auto task = [&]() {
        controller_shared->generate_text_locked(
            "prompt", "", "", [](auto s){}, [](){}, [](auto s){}
        );
    };

    // Execute
    std::thread t1(task);
    std::thread t2(task);
    t1.join();
    t2.join();

    // Assert
    // We especially care about the EXPECT_FALSE in the mock for this test.
    EXPECT_EQ(mock_ptr->call_count, 2);
}
