#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/providers/llama/llama_utils.hpp"
#include "chorus/runtime/runtime.hpp"
#include "support/runtime_test_utils.hpp"
#include "support/gtest_utils.hpp"
#include "silent_llama_log.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <condition_variable>

namespace {
using namespace Chorus;

ChorusConfig preparation_config() {
    ChorusConfig config;
    config.model.model_id = "gemma-3-270m-it-F16";
    config.model.format = ModelFormat::Gguf;
    config.model.assets.push_back({AssetRole::Weights, "tests/models/gemma-3-270m-it-F16.gguf"});
    config.provider_options["llama"] = ProviderOptionMap{{"use_gpu", false}, {"context_size", int64_t{512}}};
    config.log_level = LogLevel::Off;
    return config;
}

struct ObservedPreparation : RequestPreparation {
    std::shared_ptr<RequestPreparation> inner;
    mutable size_t counts = 0;
    mutable size_t bytes = 0;
    mutable size_t renders = 0;
    std::optional<RequestRejection> validate_request(const ChorusRequest& request) const override {
        return inner->validate_request(request);
    }
    std::variant<RenderedPrompt, RequestRejection> render_chat_prompt(
        const std::vector<ChatMessage>& messages, const std::optional<std::string>& selected, std::optional<bool> thinking
    ) const override {
        ++renders;
        return inner->render_chat_prompt(messages, selected, thinking);
    }
    std::variant<int64_t, RequestRejection> count_message_tokens(const std::string& text) const override {
        ++counts;
        bytes += text.size();
        return inner->count_message_tokens(text);
    }
};

class ObservedLlamaEngine : public LlamaEngine {
  public:
    std::shared_ptr<ObservedPreparation> observation = std::make_shared<ObservedPreparation>();
    std::shared_ptr<RequestPreparation> request_preparation() const override {
        observation->inner = LlamaEngine::request_preparation();
        return observation;
    }
};

class RuntimePreparationModelTest : public ChorusModelTest {};

TEST_F(RuntimePreparationModelTest, Literal_counts_match_independent_vendor_tokenization_and_retained_service_is_revoked) {
    const std::vector<std::string> content{"", "hello world", "é 日本語 😀", "<bos><start_of_turn>user", "abcd"};
    std::vector<int64_t> expected;
    {
        SilentLlamaLog quiet;
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        auto* model = llama_model_load_from_file("tests/models/gemma-3-270m-it-F16.gguf", params);
        ASSERT_NE(model, nullptr);
        const auto* vocabulary = llama_model_get_vocab(model);
        for (const auto& text : content) {
            std::vector<llama_token> tokens(text.size() + 32);
            const auto count = llama_tokenize(vocabulary, text.data(), static_cast<int32_t>(text.size()),
                                             tokens.data(), static_cast<int32_t>(tokens.size()), false, false);
            EXPECT_GE(count, 0);
            expected.push_back(count);
        }
        llama_model_free(model);
    }
    ChorusRuntime runtime;
    auto engine = std::make_unique<ObservedLlamaEngine>();
    auto observer = engine->observation;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), preparation_config()).ok());
    ASSERT_TRUE(runtime.capabilities()->message_token_counting);
    for (size_t i = 0; i < content.size(); ++i) {
        auto submitted = runtime.count_message_tokens(MessageContent::text(content[i]));
        ASSERT_TRUE(submitted.ok());
        auto events = drain_runtime_events(runtime);
        ASSERT_EQ(events.back().kind, RuntimeEvent::Kind::MessageTokenCount);
        ASSERT_EQ(events.back().token_count, expected[i]);
        ASSERT_EQ(events.back().request_id, submitted.request_id);
    }
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent{{std::string("ab"), std::string("cd")}}).ok());
    ASSERT_EQ(drain_runtime_events(runtime).back().token_count, expected.back());
    runtime.stop_all();
    ChorusRequest request;
    ASSERT_EQ(observer->validate_request(request)->error, ChorusError::EngineNotReady);
    ASSERT_EQ(std::get<RequestRejection>(observer->count_message_tokens("x")).error, ChorusError::EngineNotReady);
    ASSERT_EQ(std::get<RequestRejection>(observer->render_chat_prompt({}, std::nullopt, true)).error, ChorusError::EngineNotReady);
    ASSERT_FALSE(LlamaUtils::tokenize_vocabulary(nullptr, "x", false));
}

TEST_F(RuntimePreparationModelTest, Scheduler_checks_actual_render_budget_before_prefill_and_preserves_render_errors) {
    LlamaEngine engine;
    ASSERT_FALSE(engine.initialize(preparation_config(), {}, {}).has_value());
    auto service = engine.request_preparation();
    std::vector<ChatMessage> messages{{MessageRole::User, MessageContent::text("Tell me about the village.")}};
    auto error = service->render_chat_prompt(messages, "{{ raise_exception('specific failure') }}", true);
    ASSERT_TRUE(std::holds_alternative<RequestRejection>(error));
    ASSERT_NE(std::get<RequestRejection>(error).message.find("specific failure"), std::string::npos);
    auto rendered = service->render_chat_prompt(messages, std::nullopt, true);
    ASSERT_TRUE(std::holds_alternative<RenderedPrompt>(rendered));
    ASSERT_GT(std::get<RenderedPrompt>(rendered).token_count, 1);
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<ChorusSignal::Error> failure;
    size_t batches = 0;
    engine.set_batch_observer([&](const auto&) { std::lock_guard<std::mutex> lock(mutex); ++batches; });
    ChorusRequest request;
    request.id = 7;
    request.messages = messages;
    request.gen_config.max_tokens = 1;
    request.exact_prompt_budget = 1;
    request.on_event = [&](ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        if (auto* error = std::get_if<ChorusSignal::Error>(&signal.event)) failure = *error;
        cv.notify_all();
    };
    engine.submit_request(std::move(request));
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return failure.has_value(); }));
        ASSERT_EQ(failure->code, ChorusError::InvalidRequest);
        ASSERT_EQ(batches, 0U);
    }
    engine.shutdown();
}

TEST_F(RuntimePreparationModelTest, Changed_provider_render_after_preparation_fails_instead_of_bypassing_the_fit_guarantee) {
    class ChangedTemplateEngine : public LlamaEngine {
      public:
        void submit_request(ChorusRequest request) override {
            request.chat_template = "{{ 'different rendered text ' * 1000 }}";
            LlamaEngine::submit_request(std::move(request));
        }
    };
    ChorusRuntime runtime;
    ASSERT_TRUE(load_runtime(runtime, std::make_unique<ChangedTemplateEngine>(), preparation_config()).ok());
    GenerationRequest request;
    request.session_id = "changed-template";
    request.prompt = "Hello";
    request.options.max_tokens = 64;
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    ASSERT_EQ(drain_runtime_events(runtime).back().kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_TRUE(runtime.submit(request).ok());
    const auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events.back().kind, RuntimeEvent::Kind::Error);
    ASSERT_EQ(events.back().error, ChorusError::InvalidRequest);
    ASSERT_NE(events.back().text.find("prepared prompt budget"), std::string::npos);
    ASSERT_TRUE(runtime.export_conversation_history("changed-template").empty());
}

TEST_F(RuntimePreparationModelTest, Reports_separate_host_admission_completion_and_worker_work_for_review_history_shape) {
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](auto duration) { return std::chrono::duration<double, std::milli>(duration).count(); };
    for (const auto [turns, old_bytes] : std::vector<std::pair<int, size_t>>{{0, 0}, {32, 0}, {128, 0}, {256, 0}, {128, 65536}}) {
        ChorusRuntime runtime;
        auto engine = std::make_unique<ObservedLlamaEngine>();
        auto work = engine->observation;
        ASSERT_TRUE(load_runtime(runtime, std::move(engine), preparation_config()).ok());
        std::vector<ConversationMessage> history{{0, {MessageRole::System, MessageContent::text("You are the village blacksmith.")}}};
        for (int i = 0; i < turns; ++i) {
            const std::string padding = i < turns / 2 ? std::string(old_bytes, 'x') : std::string{};
            history.push_back({2 * i + 1, {MessageRole::User, MessageContent::text(padding + "Tell me about the weather in the village today.")}});
            history.push_back({2 * i + 2, {MessageRole::Assistant, MessageContent::text(padding + "The sun is shining and the village is peaceful.")}});
        }
        ASSERT_FALSE(runtime.import_conversation_history("npc", std::move(history)).has_value());
        GenerationRequest request;
        request.session_id = "npc";
        request.prompt = "What do you remember?";
        request.options.max_tokens = 64;
        request.options.temperature = 0.0f;
        for (int run = 0; run < 4; ++run) {
            const auto counts_before = work->counts;
            const auto bytes_before = work->bytes;
            const auto renders_before = work->renders;
            const auto begin = Clock::now();
            auto admitted = runtime.render_prompt(request);
            const auto admission = Clock::now();
            ASSERT_TRUE(admitted.ok());
            std::vector<RuntimeEvent> events;
            std::vector<double> poll_samples;
            const auto deadline = begin + std::chrono::seconds(10);
            while (events.empty() && Clock::now() < deadline) {
                const auto poll_begin = Clock::now();
                events = runtime.poll();
                poll_samples.push_back(milliseconds(Clock::now() - poll_begin));
                if (events.empty())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto completed = Clock::now();
            ASSERT_FALSE(events.empty());
            std::sort(poll_samples.begin(), poll_samples.end());
            ASSERT_EQ(events.back().kind, RuntimeEvent::Kind::PromptRendered);
            std::cout << "F3_PERF operation=preview turns=" << turns << " old_bytes=" << old_bytes << " run=" << run
                      << " admission_ms=" << milliseconds(admission - begin) << " completion_ms=" << milliseconds(completed - begin)
                      << " count_calls=" << work->counts - counts_before << " count_bytes=" << work->bytes - bytes_before
                      << " renders=" << work->renders - renders_before << " omitted=" << events.back().omitted_message_ids.size()
                      << " poll_p50_ms=" << poll_samples[poll_samples.size() / 2] << " poll_max_ms=" << poll_samples.back() << '\n';
        }
        const auto begin = Clock::now();
        auto admitted = runtime.submit(request);
        const auto admission = Clock::now();
        ASSERT_TRUE(admitted.ok());
        auto events = drain_runtime_events(runtime);
        const auto completed = Clock::now();
        ASSERT_EQ(events.back().kind, RuntimeEvent::Kind::Complete);
        std::cout << "F3_PERF operation=generate turns=" << turns << " old_bytes=" << old_bytes
                  << " admission_ms=" << milliseconds(admission - begin) << " completion_ms=" << milliseconds(completed - begin) << '\n';
    }
}

} // namespace
