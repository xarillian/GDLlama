#include "chorus/core/common.hpp"
#include "chorus/core/generation_config.hpp"
#include "sync_mock_engine.hpp"
#include "test_utils.hpp"

void test_chat_message_is_plain_data() {
    Chorus::ChatMessage msg{"user", "hello"};
    ASSERT_EQ(msg.role, std::string("user"));
    ASSERT_EQ(msg.content, std::string("hello"));

    Chorus::InjectedMessage injected{{"system", "it rains"}, 2};
    ASSERT_EQ(injected.depth, 2);
}

void test_chorus_request_carries_messages_and_template() {
    Chorus::ChorusRequest request;
    ASSERT_TRUE(request.messages.empty());
    ASSERT_TRUE(request.chat_template.empty());
    request.messages.push_back({"user", "hi"});
    ASSERT_EQ((int)request.messages.size(), 1);
}

void test_thinking_patch_set_and_clear() {
    Chorus::GenerationConfig base;
    Chorus::GenerationConfigPatch patch;
    patch.thinking = Chorus::ConfigPatch<bool>::set(false);
    auto with = Chorus::apply_generation_patch(base, patch);
    ASSERT_TRUE(with.thinking.has_value());
    ASSERT_TRUE(*with.thinking == false);

    Chorus::GenerationConfigPatch clear_patch;
    clear_patch.thinking = Chorus::ConfigPatch<bool>::clear();
    auto cleared = Chorus::apply_generation_patch(with, clear_patch);
    ASSERT_TRUE(!cleared.thinking.has_value());
}

void test_render_chat_prompt_defaults_to_nullopt() {
    SyncMockEngine engine; // does not override render by default
    auto rendered = engine.render_chat_prompt({{"user", "hi"}}, "", true);
    ASSERT_TRUE(!rendered.has_value());
}

void test_mock_render_counts_one_plus_words_per_message() {
    SyncMockEngine engine;
    engine.supports_render = true;
    auto rendered = engine.render_chat_prompt({{"system", "one two three"}, {"user", "four"}}, "", true);
    ASSERT_TRUE(rendered.has_value());
    // (1 + 3) + (1 + 1) = 6
    ASSERT_EQ(rendered->token_count, 6);
    ASSERT_TRUE(rendered->text.find("one two three") != std::string::npos);
}

int run_chat_type_tests() {
    std::cout << "\n--- Chat Type Tests ---" << std::endl;
    run_test("ChatTypes_message_is_plain_data", test_chat_message_is_plain_data);
    run_test("ChatTypes_request_carries_messages_and_template", test_chorus_request_carries_messages_and_template);
    run_test("ChatTypes_thinking_patch_set_and_clear", test_thinking_patch_set_and_clear);
    run_test("ChatTypes_render_chat_prompt_defaults_to_nullopt", test_render_chat_prompt_defaults_to_nullopt);
    run_test(
        "ChatTypes_mock_render_counts_one_plus_words_per_message", test_mock_render_counts_one_plus_words_per_message
    );
    return g_tests_failed > 0 ? 1 : 0;
}
