#include "chorus/runtime/runtime.hpp"
#include "chorus/runtime/runtime_preparation.hpp"
#include "support/runtime_test_utils.hpp"
#include "support/sync_mock_engine.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
using namespace Chorus;

class GatedPreparation : public RequestPreparation {
  public:
    enum class Hook { Validate, Render, Count };
    mutable std::mutex mutex;
    mutable std::condition_variable cv;
    mutable size_t calls = 0;
    mutable size_t count_calls = 0;
    mutable size_t count_bytes = 0;
    mutable size_t renders = 0;
    mutable std::set<std::thread::id> threads;
    mutable std::vector<std::string> counted;
    size_t gate_at = 0;
    mutable bool entered = false;
    bool released = false;
    int throw_kind = 0;
    std::atomic<bool> closed{false};
    mutable std::atomic<bool> gate_exited{false};
    mutable std::vector<GenerationConfig> validated_configs;
    mutable std::vector<std::optional<std::string>> validated_templates;
    struct RenderProbe {
        std::optional<std::string> chat_template;
        std::optional<bool> show_thinking;
    };
    mutable std::vector<RenderProbe> render_probes;
    std::function<std::variant<RenderedPrompt, RequestRejection>(const std::vector<ChatMessage>&)> render;
    std::optional<int64_t> count_override;
    std::optional<RequestRejection> count_rejection;

    void hook(Hook kind, const std::string& text = {}) const {
        std::unique_lock<std::mutex> lock(mutex);
        threads.insert(std::this_thread::get_id());
        ++calls;
        cv.notify_all();
        if (kind == Hook::Count) {
            ++count_calls;
            count_bytes += text.size();
            counted.push_back(text);
        }
        if (kind == Hook::Render)
            ++renders;
        if (calls == gate_at) {
            entered = true;
            cv.notify_all();
            cv.wait(lock, [&] { return released; });
            gate_exited = true;
            if (throw_kind == 1)
                throw std::runtime_error("preparation test failure");
            if (throw_kind == 2)
                throw 42;
        }
    }
    bool wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(10), [&] { return entered; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
    std::optional<RequestRejection> validate_request(const ChorusRequest& request) const override {
        hook(Hook::Validate);
        if (closed)
            return RequestRejection{ChorusError::EngineNotReady, "closed"};
        std::lock_guard<std::mutex> lock(mutex);
        validated_configs.push_back(request.gen_config);
        validated_templates.push_back(request.chat_template);
        return std::nullopt;
    }
    std::variant<RenderedPrompt, RequestRejection> render_chat_prompt(
        const std::vector<ChatMessage>& messages,
        const std::optional<std::string>& chat_template,
        std::optional<bool> show_thinking
    ) const override {
        hook(Hook::Render);
        if (closed)
            return RequestRejection{ChorusError::EngineNotReady, "closed"};
        {
            std::lock_guard<std::mutex> lock(mutex);
            render_probes.push_back({chat_template, show_thinking});
        }
        if (render)
            return render(messages);
        RenderedPrompt value;
        for (const auto& message : messages) {
            value.text += *joined_text(message.content) + "|";
            value.token_count += static_cast<int32_t>(joined_text(message.content)->size()) + 2;
        }
        return value;
    }
    std::variant<int64_t, RequestRejection> count_message_tokens(const std::string& text) const override {
        hook(Hook::Count, text);
        if (closed)
            return RequestRejection{ChorusError::EngineNotReady, "closed"};
        if (count_rejection)
            return *count_rejection;
        return count_override.value_or(static_cast<int64_t>(text.size()));
    }
};

class PreparationEngine : public SyncMockEngine {
  public:
    std::shared_ptr<GatedPreparation> service = std::make_shared<GatedPreparation>();
    bool counting = true;
    std::function<void()> on_shutdown;
    std::function<void()> on_destroy;
    std::function<void()> on_initialize;
    PreparationEngine() {
        supports_render = true;
        mock_per_request_context = 64;
    }
    ~PreparationEngine() override {
        if (on_destroy)
            on_destroy();
    }
    EngineCapabilities capabilities() const override {
        auto value = SyncMockEngine::capabilities();
        value.message_token_counting = counting;
        return value;
    }
    std::optional<InitializationFailure>
    initialize(const ChorusConfig& config, Logger logger, const InitializationControl& control) override {
        if (on_initialize)
            on_initialize();
        return SyncMockEngine::initialize(config, std::move(logger), control);
    }
    std::shared_ptr<RequestPreparation> request_preparation() const override { return service; }
    void shutdown() override {
        if (on_shutdown)
            on_shutdown();
        service->closed = true;
        SyncMockEngine::shutdown();
    }
};

struct LoadGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
    void hold() {
        std::unique_lock lock(mutex);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
    }
    bool wait_entered() {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(10), [&] { return entered; });
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        cv.notify_all();
    }
};

struct ReleaseLoadGate {
    std::shared_ptr<LoadGate> gate;
    ~ReleaseLoadGate() { gate->release(); }
};

struct ReleaseGate {
    std::shared_ptr<GatedPreparation> service;
    ~ReleaseGate() { service->release(); }
};

struct ReleaseRuntimeRetirement {
    ChorusRuntime& runtime;
    ~ReleaseRuntimeRetirement() { runtime.test_release_retirement(); }
};

GenerationRequest chat(const char* session = "npc", std::string prompt = "new") {
    GenerationRequest request;
    request.session_id = session;
    request.prompt = std::move(prompt);
    request.options.max_tokens = 0;
    return request;
}
GenerationRequest raw(std::string text) {
    GenerationRequest request;
    request.prompt = std::move(text);
    return request;
}

TEST(RuntimePreparation, Gated_hook_leaves_every_admission_poll_and_cancel_responsive_with_bounded_workers_and_queue) {
    ChorusRuntime runtime;
    runtime.test_use_preparation_workers(3);
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    ASSERT_FALSE(runtime
                     .import_conversation_history(
                         "reroll",
                         {{10, {MessageRole::User, MessageContent::text("question")}},
                          {11, {MessageRole::Assistant, MessageContent::text("reply")}}}
                     )
                     .has_value());
    auto first = runtime.submit(raw("blocked validation"));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(service->wait_entered());
    auto generation = runtime.submit(chat());
    auto regeneration = chat("reroll", "");
    auto reroll = runtime.regenerate(regeneration);
    EmbeddingRequest embedding;
    embedding.prompt = "embedding";
    auto embedded = runtime.submit(embedding);
    auto preview = runtime.render_prompt(chat());
    auto counted = runtime.count_message_tokens(MessageContent::text("draft"));
    ASSERT_TRUE(generation.ok() && reroll.ok() && embedded.ok() && preview.ok() && counted.ok());
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(observed->submitted_ids.empty());
    ASSERT_TRUE(runtime.is_loaded());
    {
        std::lock_guard<std::mutex> lock(service->mutex);
        ASSERT_LE(service->threads.size(), 3U);
        ASSERT_FALSE(service->threads.contains(std::this_thread::get_id()));
    }
    ASSERT_TRUE(runtime.cancel(preview.request_id));
    const auto cancelled_preview = runtime.poll();
    ASSERT_EQ(cancelled_preview.size(), 1U);
    ASSERT_EQ(cancelled_preview[0].error, ChorusError::Cancelled);
    ASSERT_EQ(runtime.active_request_for_session("npc"), generation.request_id);
    for (size_t i = 6; i < kPreparationCapacity; ++i)
        ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text(std::to_string(i))).ok());
    auto overflow = runtime.submit(chat("capacity"));
    ASSERT_EQ(overflow.error, ChorusError::InvalidRequest);
    ASSERT_NE(overflow.message.find("Preparation capacity"), std::string::npos);
    ASSERT_FALSE(runtime.active_request_for_session("capacity"));
    ASSERT_TRUE(runtime.export_conversation_history("capacity").empty());
    ASSERT_TRUE(runtime.cancel(first.request_id));
    ASSERT_TRUE(runtime.cancel(generation.request_id));
    ASSERT_TRUE(runtime.cancel(reroll.request_id));
    auto cancelled = drain_runtime_events(runtime, 3);
    ASSERT_EQ(cancelled.size(), 3U);
    ASSERT_TRUE(runtime.export_conversation_history("npc").empty());
    ASSERT_EQ(*joined_text(runtime.export_conversation_history("reroll").back().message.content), "reply");
    service->release();
    runtime.stop_all();
    auto remaining = drain_runtime_events(runtime, kPreparationCapacity - 4);
    ASSERT_EQ(remaining.size(), kPreparationCapacity - 4);
}

TEST(RuntimePreparation, An_earlier_request_that_prepares_slowly_still_reaches_the_engine_first) {
    ChorusRuntime runtime;
    runtime.test_use_preparation_workers(3);
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};

    const auto slow = runtime.submit(raw("slow"));
    ASSERT_TRUE(service->wait_entered());
    const auto second = runtime.submit(raw("second"));
    const auto third = runtime.submit(raw("third"));
    {
        std::unique_lock<std::mutex> lock(service->mutex);
        ASSERT_TRUE(service->cv.wait_for(lock, std::chrono::seconds(10), [&] { return service->calls >= 3; }));
    }
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(observed->submitted_ids.empty());

    service->release();
    drain_runtime_events(runtime, 3);
    EXPECT_EQ(observed->submitted_ids, (std::vector<int64_t>{slow.request_id, second.request_id, third.request_id}));
}

TEST(RuntimePreparation, Preview_freezes_history_defaults_and_source_without_occupying_or_creating_history) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    ASSERT_FALSE(runtime.import_conversation_history(
                            "npc", {{0, {MessageRole::System, MessageContent::text("old")}}}
    ).has_value());
    auto request = chat();
    auto preview = runtime.render_prompt(request);
    ASSERT_TRUE(service->wait_entered());
    ASSERT_FALSE(runtime.active_request_for_session("npc"));
    ASSERT_FALSE(runtime.edit_message("npc", 0, MessageContent::text("edited")).has_value());
    request.prompt = "changed";
    GenerationDefaults defaults;
    defaults.options.max_tokens = INT32_MAX;
    runtime.set_generation_defaults(defaults);
    ASSERT_EQ(runtime.last_turn_outcome("npc"), TurnOutcome::None);
    service->release();
    auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].request_id, preview.request_id);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_EQ(events[0].text, "old|new|");
    ASSERT_EQ(runtime.export_conversation_history("npc").size(), 1U);
    ASSERT_EQ(runtime.last_turn_outcome("npc"), TurnOutcome::None);
    auto unknown = runtime.render_prompt(chat("unknown"));
    ASSERT_TRUE(unknown.ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(runtime.list_conversations(), (std::vector<SessionId>{"npc"}));
}

TEST(RuntimePreparation, Lazy_counts_reuse_nodes_invalidate_edits_and_verify_extra_omission_on_submitted_messages) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->mock_per_request_context = 24;
    auto* observed = engine.get();
    auto service = engine->service;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    std::vector<ConversationMessage> history{{0, {MessageRole::System, MessageContent::text("S")}}};
    for (int i = 0; i < 128; ++i) {
        history.push_back({1 + 2 * i, {MessageRole::User, MessageContent::text("user")}});
        history.push_back({2 + 2 * i, {MessageRole::Assistant, MessageContent::text("reply")}});
    }
    ASSERT_FALSE(runtime.import_conversation_history("npc", history).has_value());
    auto request = chat();
    auto preview = runtime.render_prompt(request);
    auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events.back().kind, RuntimeEvent::Kind::PromptRendered);
    const size_t cold = service->count_calls;
    ASSERT_LT(cold, 12U);
    ASSERT_GT(service->renders, 1U);
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    auto repeated = drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, cold);
    ASSERT_EQ(repeated.back().text, events.back().text);
    ASSERT_FALSE(runtime.edit_message("npc", 0, MessageContent::text("T")).has_value());
    auto edited_preview = runtime.render_prompt(request);
    ASSERT_TRUE(edited_preview.ok());
    const auto edited_events = drain_runtime_events(runtime);
    ASSERT_EQ(edited_events.size(), 1U);
    ASSERT_EQ(edited_events.front().request_id, edited_preview.request_id);
    ASSERT_EQ(edited_events.front().kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_EQ(service->count_calls, cold + 1);
    auto generated = runtime.submit(request);
    ASSERT_TRUE(generated.ok());
    auto submitted = drain_runtime_events(runtime);
    int32_t actual_count = 0;
    std::string submitted_text;
    for (const auto& message : observed->last_messages) {
        const auto content = joined_text(message.content);
        ASSERT_TRUE(content.has_value());
        actual_count += static_cast<int32_t>(content->size()) + 2;
        submitted_text += *content + "|";
    }
    ASSERT_EQ(submitted_text, edited_events.front().text);
    ASSERT_LE(actual_count, 24);
    ASSERT_EQ(observed->last_messages.front().role, MessageRole::System);
    ASSERT_FALSE(submitted.empty());
    ASSERT_EQ(submitted.front().request_id, generated.request_id);
    ASSERT_EQ(submitted.front().kind, RuntimeEvent::Kind::HistoryTruncated);
    ASSERT_EQ(submitted.front().omitted_message_ids, edited_events.front().omitted_message_ids);
    ASSERT_FALSE(submitted.front().omitted_message_ids.empty());
    ASSERT_EQ(submitted.front().omitted_message_ids.front(), 1);
    ASSERT_EQ(submitted.front().omitted_message_ids.size() % 2, 0U);
}

TEST(RuntimePreparation, Nonmonotonic_fallback_recovers_a_larger_candidate_when_minimal_fails) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->mock_per_request_context = 8;
    auto service = engine->service;
    service->render = [](const std::vector<ChatMessage>& messages) {
        return RenderedPrompt{std::to_string(messages.size()), messages.size() == 3 ? 4 : 99};
    };
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ASSERT_FALSE(runtime
                     .import_conversation_history(
                         "npc",
                         {{0, {MessageRole::User, MessageContent::text("oversized-old-content")}},
                          {1, {MessageRole::Assistant, MessageContent::text("reply")}}}
                     )
                     .has_value());
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_EQ(events[0].text, "3");
    ASSERT_TRUE(events[0].omitted_message_ids.empty());
    ASSERT_EQ(service->renders, 2U);
}

TEST(RuntimePreparation, Mandatory_estimate_overflow_does_not_reject_a_fitting_nonadditive_render) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->service->count_override = INT64_MAX;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ASSERT_FALSE(runtime.import_conversation_history(
        "npc",
        {{0, {MessageRole::System, MessageContent::text("S")}}, {1, {MessageRole::System, MessageContent::text("T")}}}
    ));
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    const auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_EQ(events[0].text, "S|T|new|");
}

TEST(RuntimePreparation, Tokenizer_and_renderer_failures_are_errors_not_zero_or_unsupported_fallback) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    engine->counting = false;
    engine->service->render = [](const auto&) -> std::variant<RenderedPrompt, RequestRejection> {
        return RequestRejection{ChorusError::Tokenize, "rendered tokenization failed"};
    };
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ASSERT_TRUE(runtime.submit(chat()).ok());
    auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].error, ChorusError::Tokenize);
    ASSERT_TRUE(observed->submitted_ids.empty());
    ASSERT_TRUE(runtime.export_conversation_history("npc").empty());
    auto replacement = std::make_unique<PreparationEngine>();
    replacement->service->count_rejection = RequestRejection{ChorusError::Tokenize, "raw tokenization failed"};
    ASSERT_TRUE(load_runtime(runtime, std::move(replacement), {}).ok());
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text("text")).ok());
    events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::Error);
    ASSERT_EQ(events[0].error, ChorusError::Tokenize);
}

TEST(RuntimePreparation, Counts_join_parts_and_bound_arbitrary_content_cache_and_reset_on_reload) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    MessageContent content{{std::string("ab"), std::string("cd")}};
    ASSERT_TRUE(runtime.count_message_tokens(content).ok());
    auto first = drain_runtime_events(runtime);
    ASSERT_EQ(first[0].token_count, 4);
    ASSERT_EQ(service->counted.back(), "abcd");
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text("abcd")).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, 1U);
    ASSERT_TRUE(runtime.count_message_tokens({}).ok());
    ASSERT_EQ(drain_runtime_events(runtime)[0].token_count, 0);
    for (size_t i = 0; i < kContentCountCacheEntries; ++i) {
        ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text(std::to_string(i))).ok());
        drain_runtime_events(runtime);
    }
    auto before = service->count_calls;
    ASSERT_TRUE(runtime.count_message_tokens(content).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, before + 1);
    std::string oversized(kContentCountCacheBytes + 1, 'x');
    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text(oversized)).ok());
        drain_runtime_events(runtime);
    }
    ASSERT_EQ(service->count_calls, before + 3);
    for (char fill : {'a', 'b', 'c'}) {
        ASSERT_TRUE(
            runtime.count_message_tokens(MessageContent::text(std::string(kContentCountCacheBytes / 2, fill))).ok()
        );
        drain_runtime_events(runtime);
    }
    before = service->count_calls;
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text(std::string(kContentCountCacheBytes / 2, 'a'))).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, before + 1);
    auto replacement = std::make_unique<PreparationEngine>();
    auto fresh = replacement->service;
    ASSERT_TRUE(load_runtime(runtime, std::move(replacement), {}).ok());
    ASSERT_TRUE(service->closed);
    ASSERT_TRUE(runtime.count_message_tokens(content).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(fresh->count_calls, 1U);
}

class PreparationFailure : public ::testing::TestWithParam<int> {};
TEST_P(PreparationFailure, Unexpected_exception_fails_ready_running_and_queued_once_and_closes_admission) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 2;
    service->throw_kind = GetParam();
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    auto ready = runtime.submit(raw("ready"));
    auto running = runtime.submit(raw("running"));
    auto queued = runtime.submit(chat());
    ASSERT_TRUE(service->wait_entered());
    service->release();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (runtime.is_loaded() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_FALSE(runtime.is_loaded());
    const auto events = drain_runtime_events(runtime, 3);
    std::set<RequestId> ids;
    size_t failures = 0;
    for (const auto& event : events) {
        if (event.kind == RuntimeEvent::Kind::EngineFailed) {
            ++failures;
            continue;
        }
        ASSERT_EQ(event.kind, RuntimeEvent::Kind::Error);
        ASSERT_TRUE(ids.insert(event.request_id).second);
    }
    ASSERT_EQ(ids, (std::set<RequestId>{ready.request_id, running.request_id, queued.request_id}));
    ASSERT_EQ(failures, 1U);
    ASSERT_TRUE(observed->submitted_ids.empty());
    ASSERT_TRUE(runtime.export_conversation_history("npc").empty());
    ASSERT_EQ(runtime.submit(raw("late")).error, ChorusError::EngineNotReady);
    ASSERT_TRUE(runtime.poll().empty());
}
TEST_P(PreparationFailure, Worker_failure_cancels_provider_owned_work_without_losing_runtime_owned_jobs) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    engine->hold_requests = true;
    engine->emit_cancelled_on_cancel = true;
    auto service = engine->service;
    service->gate_at = 2;
    service->throw_kind = GetParam();
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    auto active = runtime.submit(raw("provider active"));
    forward_runtime_until(runtime, [&] { return observed->submitted_ids.size() == 1; });
    auto running = runtime.submit(raw("preparing"));
    auto queued = runtime.count_message_tokens(MessageContent::text("queued"));
    ASSERT_TRUE(service->wait_entered());
    service->release();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (runtime.is_loaded() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const auto events = drain_runtime_events(runtime, 3);
    std::set<RequestId> ids;
    for (const auto& event : events) {
        if (event.kind == RuntimeEvent::Kind::EngineFailed)
            continue;
        ASSERT_EQ(event.kind, RuntimeEvent::Kind::Error);
        ASSERT_TRUE(ids.insert(event.request_id).second);
    }
    ASSERT_EQ(ids, (std::set<RequestId>{active.request_id, running.request_id, queued.request_id}));
    ASSERT_EQ(observed->submitted_ids, (std::vector<RequestId>{active.request_id}));
    ASSERT_TRUE(runtime.poll().empty());
}
INSTANTIATE_TEST_SUITE_P(StandardAndNonstandard, PreparationFailure, ::testing::Values(1, 2));

TEST(RuntimePreparation, Provider_death_drains_gated_and_queued_preparation_without_waiting_for_the_hook) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text("gated")).ok());
    ASSERT_TRUE(service->wait_entered());
    ASSERT_TRUE(runtime.submit(chat()).ok());
    observed->die();
    const auto events = drain_runtime_events(runtime, 2);
    ASSERT_EQ(events.size(), 3U);
    ASSERT_EQ(events.back().kind, RuntimeEvent::Kind::EngineFailed);
    ASSERT_FALSE(service->gate_exited);
    ASSERT_TRUE(runtime.export_conversation_history("npc").empty());
    service->release();
    runtime.stop_all();
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(RuntimePreparation, Cancel_ready_work_never_forwards_and_buffered_auxiliary_success_keeps_its_terminal) {
    ChorusRuntime runtime;
    runtime.test_use_preparation_workers(1);
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 3;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    const auto success = runtime.count_message_tokens(MessageContent::text("count"));
    const auto ready = runtime.submit(raw("ready"));
    const auto gated = runtime.submit(raw("gated"));
    ASSERT_TRUE(service->wait_entered());
    ASSERT_TRUE(runtime.cancel(success.request_id));
    ASSERT_TRUE(runtime.cancel(ready.request_id));
    auto events = drain_runtime_events(runtime, 2);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::MessageTokenCount);
    ASSERT_EQ(events[0].request_id, success.request_id);
    ASSERT_EQ(events[1].error, ChorusError::Cancelled);
    ASSERT_EQ(events[1].request_id, ready.request_id);
    ASSERT_TRUE(observed->submitted_ids.empty());
    ASSERT_TRUE(runtime.cancel(gated.request_id));
    drain_runtime_events(runtime);
    service->release();
    runtime.stop_all();
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(RuntimePreparation, Gated_render_and_count_cancel_without_waiting_and_discard_late_success) {
    for (bool count : {false, true}) {
        ChorusRuntime runtime;
        auto engine = std::make_unique<PreparationEngine>();
        engine->counting = count;
        auto service = engine->service;
        service->gate_at = 1;
        ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
        ReleaseGate release{service};
        auto accepted =
            count ? runtime.count_message_tokens(MessageContent::text("draft")) : runtime.render_prompt(chat());
        ASSERT_TRUE(accepted.ok());
        ASSERT_TRUE(service->wait_entered());
        ASSERT_TRUE(runtime.poll().empty());
        ASSERT_TRUE(runtime.cancel(accepted.request_id));
        auto events = runtime.poll();
        ASSERT_EQ(events.size(), 1U);
        ASSERT_EQ(events[0].error, ChorusError::Cancelled);
        ASSERT_FALSE(runtime.is_request_active(accepted.request_id));
        service->release();
        runtime.stop_all();
        ASSERT_TRUE(runtime.poll().empty());
        ASSERT_TRUE(runtime.list_conversations().empty());
    }
}

TEST(RuntimePreparation, Batch_defaults_are_frozen_before_a_gated_worker_can_observe_later_defaults) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    GenerationDefaults defaults;
    defaults.options.temperature = 0.25f;
    runtime.set_generation_defaults(defaults);
    auto results = runtime.submit_batch(std::vector<GenerationRequest>{raw("first"), raw("second"), raw("third")});
    ASSERT_TRUE(service->wait_entered());
    defaults.options.temperature = 0.75f;
    runtime.set_generation_defaults(defaults);
    service->release();
    auto events = drain_runtime_events(runtime, 3);
    ASSERT_EQ(events.size(), 3U);
    ASSERT_EQ(service->validated_configs.size(), 3U);
    for (const auto& config : service->validated_configs)
        ASSERT_EQ(config.temperature, 0.25f);
}

TEST(RuntimePreparation, Preview_and_generation_preserve_optional_chat_controls) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    auto request = chat();
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    ASSERT_EQ(drain_runtime_events(runtime).back().kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_FALSE(service->render_probes.back().show_thinking);
    ASSERT_FALSE(service->render_probes.back().chat_template);
    ASSERT_FALSE(service->validated_configs.back().show_thinking);
    ASSERT_FALSE(service->validated_templates.back());
    GenerationDefaults defaults;
    defaults.chat_template = "host";
    defaults.options.show_thinking = false;
    runtime.set_generation_defaults(defaults);
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->render_probes.back().show_thinking, false);
    ASSERT_EQ(service->render_probes.back().chat_template, "host");
    request.options.show_thinking = true;
    request.chat_template = "request";
    ASSERT_TRUE(runtime.submit(request).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->render_probes.back().show_thinking, true);
    ASSERT_EQ(service->render_probes.back().chat_template, "request");
    ASSERT_EQ(service->validated_configs.back().show_thinking, true);
    ASSERT_EQ(service->validated_templates.back(), "request");
    ASSERT_EQ(observed->last_config.show_thinking, true);
    ASSERT_EQ(observed->last_chat_template, "request");
    request.session_id = "cleared";
    request.options.show_thinking.reset();
    request.chat_template.reset();
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->render_probes.back().show_thinking, false);
    ASSERT_EQ(service->render_probes.back().chat_template, "host");
    request.chat_template = "";
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->render_probes.back().chat_template, "");
    ASSERT_EQ(service->validated_templates.back(), "");
}

TEST(RuntimePreparation, Admitted_request_and_batch_keep_earlier_defaults_after_replacement) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    GenerationDefaults defaults;
    defaults.options.max_tokens = 0;
    defaults.options.stop = std::vector<std::string>{"old"};
    defaults.options.show_thinking = false;
    defaults.chat_template = "old template";
    runtime.set_generation_defaults(defaults);
    auto first = chat("first");
    first.options.max_tokens.reset();
    auto accepted = runtime.submit(first);
    ASSERT_TRUE(accepted.ok());
    ASSERT_TRUE(service->wait_entered());
    auto second = chat("second");
    second.options.max_tokens.reset();
    auto third = chat("third");
    third.options.max_tokens = 3;
    auto batch = runtime.submit_batch(std::vector<GenerationRequest>{second, third});
    ASSERT_EQ(batch.size(), 2U);
    ASSERT_TRUE(batch[0].ok() && batch[1].ok());
    defaults.options.max_tokens = 12;
    defaults.options.stop = std::vector<std::string>{"new"};
    defaults.options.show_thinking = true;
    defaults.chat_template = "new template";
    runtime.set_generation_defaults(defaults);
    service->release();
    auto events = drain_runtime_events(runtime, 3);
    ASSERT_EQ(events.size(), 3U);
    ASSERT_EQ(service->validated_configs.size(), 3U);
    ASSERT_EQ(service->render_probes.size(), 3U);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(service->validated_configs[i].stop, (std::vector<std::string>{"old"}));
        ASSERT_EQ(service->validated_configs[i].show_thinking, false);
        ASSERT_EQ(service->validated_templates[i], "old template");
        ASSERT_EQ(service->render_probes[i].chat_template, "old template");
        ASSERT_EQ(service->render_probes[i].show_thinking, false);
    }
    std::vector<std::optional<int32_t>> budgets;
    for (const auto& config : service->validated_configs)
        budgets.push_back(config.max_tokens);
    std::ranges::sort(budgets);
    ASSERT_EQ(budgets, (std::vector<std::optional<int32_t>>{0, 0, 3}));
    ASSERT_EQ(observed->last_chat_template, "old template");
    ASSERT_EQ(observed->last_config.stop, (std::vector<std::string>{"old"}));
    auto next = chat("fourth");
    next.options.max_tokens.reset();
    ASSERT_TRUE(runtime.submit(next).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(observed->last_config.max_tokens, 12);
    ASSERT_EQ(observed->last_chat_template, "new template");
}

TEST(RuntimePreparation, Progress_is_coalesced_and_invalid_samples_do_not_publish_readiness) {
    class ProgressEngine : public PreparationEngine {
      public:
        std::shared_ptr<std::atomic<bool>> reported;
        std::optional<InitializationFailure>
        initialize(const ChorusConfig& config, Logger logger, const InitializationControl& control) override {
            for (int i = 0; i < 5000; ++i)
                control.on_progress({LoadPhase::LoadingModel, 0.5f});
            control.on_progress({LoadPhase::LoadingModel, 0.25f});
            control.on_progress({LoadPhase::LoadingModel, std::numeric_limits<float>::quiet_NaN()});
            control.on_progress({LoadPhase::InitializingEngine, std::nullopt});
            control.on_progress({LoadPhase::LoadingModel, 1.0f});
            reported->store(true);
            return PreparationEngine::initialize(config, std::move(logger), control);
        }
    };
    ChorusRuntime runtime;
    auto reported = std::make_shared<std::atomic<bool>>(false);
    auto engine = std::make_unique<ProgressEngine>();
    engine->reported = reported;
    auto admitted = runtime.load_engine(std::move(engine), {});
    ASSERT_TRUE(admitted.ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!reported->load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_TRUE(reported->load());
    auto observed = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_TRUE(observed.ok());
    ASSERT_EQ(observed.events.size(), 2U);
    ASSERT_EQ(observed.events[0].kind, RuntimeEvent::Kind::ModelLoadProgress);
    ASSERT_EQ(observed.events[0].load_progress->phase, LoadPhase::InitializingEngine);
    ASSERT_FALSE(observed.events[0].load_progress->fraction);
    ASSERT_EQ(observed.events[1].kind, RuntimeEvent::Kind::ModelLoaded);
    auto logs = runtime.poll_logs();
    ASSERT_EQ(logs.size(), 1U);
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(RuntimePreparation, In_flight_progress_drains_across_polls_without_regressing_or_replaying) {
    class ProgressEngine : public PreparationEngine {
      public:
        std::shared_ptr<LoadGate> first;
        std::shared_ptr<LoadGate> second;
        std::optional<InitializationFailure>
        initialize(const ChorusConfig& config, Logger logger, const InitializationControl& control) override {
            control.on_progress({LoadPhase::LoadingModel, 0.4f});
            first->hold();
            control.on_progress({LoadPhase::LoadingModel, 0.2f});
            control.on_progress({LoadPhase::LoadingModel, 0.7f});
            second->hold();
            control.on_progress({LoadPhase::LoadingModel, 0.6f});
            return PreparationEngine::initialize(config, std::move(logger), control);
        }
    };
    ChorusRuntime runtime;
    auto first = std::make_shared<LoadGate>();
    auto second = std::make_shared<LoadGate>();
    ReleaseLoadGate release_first{first};
    ReleaseLoadGate release_second{second};
    auto engine = std::make_unique<ProgressEngine>();
    engine->first = first;
    engine->second = second;
    auto admitted = runtime.load_engine(std::move(engine), {});
    ASSERT_TRUE(admitted.ok());
    ASSERT_TRUE(first->wait_entered());
    auto early = runtime.poll();
    ASSERT_EQ(early.size(), 1U);
    ASSERT_EQ(early[0].kind, RuntimeEvent::Kind::ModelLoadProgress);
    ASSERT_EQ(early[0].load_id, admitted.load_id);
    ASSERT_EQ(early[0].load_progress->fraction, 0.4f);
    ASSERT_TRUE(runtime.poll().empty());
    first->release();
    ASSERT_TRUE(second->wait_entered());
    auto later = runtime.poll();
    ASSERT_EQ(later.size(), 1U);
    ASSERT_EQ(later[0].kind, RuntimeEvent::Kind::ModelLoadProgress);
    ASSERT_EQ(later[0].load_progress->fraction, 0.7f);
    ASSERT_TRUE(runtime.poll().empty());
    second->release();
    auto terminal = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_TRUE(terminal.ok());
    ASSERT_EQ(terminal.events.size(), 1U);
    ASSERT_EQ(terminal.events[0].kind, RuntimeEvent::Kind::ModelLoaded);
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(RuntimePreparation, Initialization_is_off_thread_and_cancellation_wins_before_return) {
    ChorusRuntime runtime;
    auto gate = std::make_shared<LoadGate>();
    ReleaseLoadGate release{gate};
    auto engine = std::make_unique<PreparationEngine>();
    engine->on_initialize = [gate] { gate->hold(); };
    auto config = ChorusConfig{};
    config.model.model_id = "original";
    auto admitted = runtime.load_engine(std::move(engine), config);
    ASSERT_TRUE(admitted.ok());
    ASSERT_TRUE(gate->wait_entered());
    config.model.model_id = "modified";
    ASSERT_FALSE(runtime.is_loaded());
    ASSERT_EQ(runtime.active_load_id(), admitted.load_id);
    ASSERT_EQ(runtime.submit(raw("too soon")).error, ChorusError::EngineNotReady);
    ASSERT_EQ(runtime.load_engine(std::make_unique<PreparationEngine>(), {}).error, ChorusError::InvalidRequest);
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(runtime.cancel_load(admitted.load_id));
    gate->release();
    auto cancelled = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_EQ(cancelled.error, ChorusError::Cancelled);
    ASSERT_EQ(cancelled.events.back().model_id, "original");
    ASSERT_EQ(cancelled.events.back().kind, RuntimeEvent::Kind::ModelLoadFailed);
    ASSERT_FALSE(runtime.cancel_load(cancelled.load_id));
    ASSERT_TRUE(runtime.poll().empty());
    auto retry = load_runtime(runtime, std::make_unique<PreparationEngine>(), {});
    ASSERT_TRUE(retry.ok());
    ASSERT_TRUE(runtime.is_loaded());
}

TEST(RuntimePreparation, Cancellation_after_candidate_is_parked_prevents_publication) {
    ChorusRuntime runtime;
    auto admitted = runtime.load_engine(std::make_unique<PreparationEngine>(), {});
    ASSERT_TRUE(admitted.ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!runtime.test_load_parked(admitted.load_id) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_TRUE(runtime.test_load_parked(admitted.load_id));
    ASSERT_FALSE(runtime.is_loaded());
    ASSERT_EQ(runtime.active_load_id(), admitted.load_id);
    ASSERT_TRUE(runtime.cancel_load(admitted.load_id));
    auto observed = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_EQ(observed.error, ChorusError::Cancelled);
    ASSERT_EQ(observed.events.back().kind, RuntimeEvent::Kind::ModelLoadFailed);
    ASSERT_FALSE(runtime.is_loaded());
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(RuntimePreparation, Cancellation_accepted_before_provider_failure_wins_and_later_cancel_cannot_rewrite_failure) {
    ChorusRuntime runtime;
    auto gate = std::make_shared<LoadGate>();
    ReleaseLoadGate release{gate};
    auto engine = std::make_unique<PreparationEngine>();
    engine->on_initialize = [gate] { gate->hold(); };
    engine->fail_initialize_with = ChorusError::ModelLoad;
    auto admitted = runtime.load_engine(std::move(engine), {});
    ASSERT_TRUE(admitted.ok());
    ASSERT_TRUE(gate->wait_entered());
    ASSERT_TRUE(runtime.cancel_load(admitted.load_id));
    ASSERT_TRUE(runtime.cancel_load(admitted.load_id));
    gate->release();
    auto cancelled = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_EQ(cancelled.error, ChorusError::Cancelled);
    ASSERT_FALSE(runtime.cancel_load(cancelled.load_id));
    auto next = std::make_unique<PreparationEngine>();
    next->fail_initialize_with = ChorusError::ModelLoad;
    auto failed_admission = runtime.load_engine(std::move(next), {});
    ASSERT_TRUE(failed_admission.ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!runtime.test_load_committed(failed_admission.load_id) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_TRUE(runtime.test_load_committed(failed_admission.load_id));
    ASSERT_EQ(runtime.active_load_id(), failed_admission.load_id);
    ASSERT_FALSE(runtime.cancel_load(failed_admission.load_id));
    ASSERT_EQ(runtime.load_engine(std::make_unique<PreparationEngine>(), {}).error, ChorusError::InvalidRequest);
    auto failed = wait_load_terminal(runtime, std::move(failed_admission));
    ASSERT_EQ(failed.error, ChorusError::ModelLoad);
    ASSERT_EQ(failed.events.back().kind, RuntimeEvent::Kind::ModelLoadFailed);
    ASSERT_EQ(failed.events.back().error, ChorusError::ModelLoad);
    ASSERT_FALSE(runtime.cancel_load(failed.load_id));
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(load_runtime(runtime, std::make_unique<PreparationEngine>(), {}).ok());
}

TEST(RuntimePreparation, Stop_fences_unpublished_candidate_and_preserves_the_load_terminal) {
    ChorusRuntime runtime;
    auto gate = std::make_shared<LoadGate>();
    ReleaseLoadGate release{gate};
    auto engine = std::make_unique<PreparationEngine>();
    engine->on_initialize = [gate] { gate->hold(); };
    auto admitted = runtime.load_engine(std::move(engine), {});
    ASSERT_TRUE(admitted.ok());
    ASSERT_TRUE(gate->wait_entered());
    ASSERT_TRUE(runtime.cancel_load(admitted.load_id));
    gate->release();
    runtime.stop_all();
    runtime.stop_all();
    ASSERT_FALSE(runtime.is_loaded());
    auto terminal = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_EQ(terminal.error, ChorusError::Cancelled);
    ASSERT_EQ(terminal.events.back().kind, RuntimeEvent::Kind::ModelLoadFailed);
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(load_runtime(runtime, std::make_unique<PreparationEngine>(), {}).ok());
}

TEST(RuntimePreparation, Partial_initialization_failures_fence_resources_before_terminal_and_recover) {
    enum class Failure { StandardException, NonstandardException, FalseReadiness, MissingPreparation, WorkerStart };
    class Candidate : public PreparationEngine {
      public:
        explicit Candidate(Failure failure) : failure(failure) {}
        Failure failure;
        std::optional<InitializationFailure>
        initialize(const ChorusConfig& config, Logger logger, const InitializationControl& control) override {
            auto result = PreparationEngine::initialize(config, std::move(logger), control);
            if (failure == Failure::StandardException)
                throw std::runtime_error("partial initialization failed");
            return result;
        }
        bool is_initialized() const override {
            return failure != Failure::FalseReadiness && PreparationEngine::is_initialized();
        }
        std::shared_ptr<RequestPreparation> request_preparation() const override {
            return failure == Failure::MissingPreparation ? nullptr : PreparationEngine::request_preparation();
        }
        std::optional<LoadedModelInfo> loaded_model_info() const override {
            if (failure == Failure::NonstandardException)
                throw 42;
            return PreparationEngine::loaded_model_info();
        }
    };
    struct Cleanup {
        std::atomic<int> shutdowns{0};
        std::atomic<int> destructions{0};
        std::atomic<bool> closed_before_destruction{false};
    };
    ChorusRuntime runtime;
    for (auto failure :
         {Failure::StandardException,
          Failure::NonstandardException,
          Failure::FalseReadiness,
          Failure::MissingPreparation,
          Failure::WorkerStart}) {
        SCOPED_TRACE(static_cast<int>(failure));
        auto cleanup = std::make_shared<Cleanup>();
        auto candidate = std::make_unique<Candidate>(failure);
        std::weak_ptr<GatedPreparation> service = candidate->service;
        candidate->on_shutdown = [cleanup] { ++cleanup->shutdowns; };
        candidate->on_destroy = [cleanup, service] {
            if (auto retained = service.lock())
                cleanup->closed_before_destruction = retained->closed.load();
            ++cleanup->destructions;
        };
        if (failure == Failure::WorkerStart)
            runtime.test_fail_next_preparation_worker_start();
        auto admitted = runtime.load_engine(std::move(candidate), {});
        ASSERT_TRUE(admitted.ok());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!runtime.test_load_committed(admitted.load_id) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        ASSERT_TRUE(runtime.test_load_committed(admitted.load_id));
        ASSERT_EQ(runtime.active_load_id(), admitted.load_id);
        ASSERT_EQ(cleanup->shutdowns.load(), 1);
        ASSERT_EQ(cleanup->destructions.load(), 1);
        ASSERT_TRUE(cleanup->closed_before_destruction.load());
        ASSERT_TRUE(service.expired());
        auto observed = wait_load_terminal(runtime, std::move(admitted));
        ASSERT_EQ(
            observed.error,
            failure == Failure::FalseReadiness || failure == Failure::MissingPreparation ? ChorusError::EngineNotReady
                                                                                         : ChorusError::Unknown
        );
        ASSERT_EQ(observed.events.back().kind, RuntimeEvent::Kind::ModelLoadFailed);
        ASSERT_EQ(
            std::count_if(
                observed.events.begin(),
                observed.events.end(),
                [](const auto& event) {
                    return event.kind == RuntimeEvent::Kind::ModelLoadFailed ||
                           event.kind == RuntimeEvent::Kind::ModelLoaded;
                }
            ),
            1
        );
        ASSERT_TRUE(std::none_of(observed.events.begin(), observed.events.end(), [](const auto& event) {
            return event.kind == RuntimeEvent::Kind::EngineFailed;
        }));
        ASSERT_FALSE(observed.events.back().text.empty());
        if (failure == Failure::StandardException)
            ASSERT_EQ(observed.events.back().text, "partial initialization failed");
        ASSERT_FALSE(runtime.is_loaded());
        ASSERT_TRUE(runtime.poll().empty());
        ASSERT_TRUE(load_runtime(runtime, std::make_unique<PreparationEngine>(), {}).ok());
        auto request = runtime.submit(raw("after failed load"));
        ASSERT_TRUE(request.ok());
        ASSERT_EQ(drain_runtime_events(runtime).back().kind, RuntimeEvent::Kind::Complete);
    }
}

TEST(RuntimePreparation, Failed_load_setup_and_exhausted_identity_preserve_the_loaded_engine) {
    ChorusRuntime failed_start;
    failed_start.test_fail_next_worker_start();
    auto rejected_start = failed_start.load_engine(std::make_unique<PreparationEngine>(), {});
    ASSERT_EQ(rejected_start.error, ChorusError::Unknown);
    ASSERT_EQ(rejected_start.load_id, -1);
    ASSERT_FALSE(failed_start.active_load_id());
    ASSERT_TRUE(load_runtime(failed_start, std::make_unique<PreparationEngine>(), {}).ok());
    auto current = failed_start.submit(raw("still usable"));
    ASSERT_TRUE(current.ok());
    failed_start.test_fail_next_load_setup();
    auto rejected_setup = failed_start.load_engine(std::make_unique<PreparationEngine>(), {});
    ASSERT_EQ(rejected_setup.error, ChorusError::Unknown);
    ASSERT_EQ(rejected_setup.load_id, -1);
    ASSERT_FALSE(failed_start.active_load_id());
    ASSERT_TRUE(failed_start.is_loaded());
    ASSERT_EQ(drain_runtime_events(failed_start)[0].request_id, current.request_id);
    failed_start.test_exhaust_load_ids();
    auto exhausted = failed_start.load_engine(std::make_unique<PreparationEngine>(), {});
    ASSERT_EQ(exhausted.error, ChorusError::InvalidRequest);
    ASSERT_EQ(exhausted.load_id, -1);
    ASSERT_FALSE(failed_start.active_load_id());
    ASSERT_TRUE(failed_start.is_loaded());
    ASSERT_TRUE(failed_start.submit(raw("after rejection")).ok());
    ASSERT_EQ(drain_runtime_events(failed_start).back().kind, RuntimeEvent::Kind::Complete);
}

TEST(RuntimePreparation, Parked_candidate_that_loses_health_is_cleaned_up_before_failure_terminal) {
    class FailingCandidate : public PreparationEngine {
      public:
        std::shared_ptr<std::atomic<bool>> healthy;
        bool is_initialized() const override { return PreparationEngine::is_initialized() && healthy->load(); }
    };
    int shutdowns = 0;
    ChorusRuntime runtime;
    auto candidate = std::make_unique<FailingCandidate>();
    auto healthy = std::make_shared<std::atomic<bool>>(true);
    candidate->healthy = healthy;
    candidate->shutdown_count_sink = &shutdowns;
    auto admitted = runtime.load_engine(std::move(candidate), {});
    ASSERT_TRUE(admitted.ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!runtime.test_load_parked(admitted.load_id) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_TRUE(runtime.test_load_parked(admitted.load_id));
    healthy->store(false);
    auto observed = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_EQ(observed.error, ChorusError::EngineNotReady);
    ASSERT_EQ(observed.events.back().kind, RuntimeEvent::Kind::ModelLoadFailed);
    ASSERT_EQ(shutdowns, 1);
    ASSERT_FALSE(runtime.is_loaded());
    ASSERT_TRUE(runtime.poll().empty());
    ASSERT_TRUE(load_runtime(runtime, std::make_unique<PreparationEngine>(), {}).ok());
}

TEST(RuntimePreparation, Acceptance_closes_old_preparation_before_worker_retires_it) {
    ChorusRuntime runtime;
    runtime.test_use_preparation_workers(1);
    auto old = std::make_unique<PreparationEngine>();
    auto service = old->service;
    service->gate_at = 1;
    ASSERT_TRUE(load_runtime(runtime, std::move(old), {}).ok());
    ReleaseGate release_service{service};
    auto first = runtime.count_message_tokens(MessageContent::text("running"));
    ASSERT_TRUE(service->wait_entered());
    auto queued = runtime.count_message_tokens(MessageContent::text("queued"));
    ASSERT_TRUE(first.ok() && queued.ok());
    runtime.test_hold_next_retirement();
    ReleaseRuntimeRetirement release_retirement{runtime};
    auto admitted = runtime.load_engine(std::make_unique<PreparationEngine>(), {});
    ASSERT_TRUE(admitted.ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!runtime.test_retirement_held() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool held = runtime.test_retirement_held();
    if (!held) {
        service->release();
        runtime.test_release_retirement();
    }
    ASSERT_TRUE(held);
    auto progress = runtime.poll();
    ASSERT_EQ(progress.back().kind, RuntimeEvent::Kind::ModelLoadProgress);
    ASSERT_EQ(progress.back().load_progress->phase, LoadPhase::ReleasingEngine);
    service->release();
    {
        std::lock_guard lock(service->mutex);
        ASSERT_EQ(service->count_calls, 1U);
    }
    runtime.test_release_retirement();
    auto observed = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_TRUE(observed.ok());
    std::erase_if(observed.events, [](const auto& event) {
        return event.kind == RuntimeEvent::Kind::ModelLoadProgress;
    });
    ASSERT_EQ(observed.events.size(), 3U);
    ASSERT_EQ(observed.events[0].error, ChorusError::Cancelled);
    ASSERT_EQ(observed.events[1].error, ChorusError::Cancelled);
    ASSERT_EQ(
        std::set<RequestId>({observed.events[0].request_id, observed.events[1].request_id}),
        std::set<RequestId>({first.request_id, queued.request_id})
    );
    ASSERT_EQ(observed.events[2].kind, RuntimeEvent::Kind::ModelLoaded);
}

TEST(RuntimePreparation, Replacement_admission_does_not_wait_for_old_shutdown_or_start_candidate_early) {
    ChorusRuntime runtime;
    auto old = std::make_unique<PreparationEngine>();
    auto gate = std::make_shared<LoadGate>();
    ReleaseLoadGate release{gate};
    old->on_shutdown = [gate] { gate->hold(); };
    ASSERT_TRUE(load_runtime(runtime, std::move(old), {}).ok());
    auto next = std::make_unique<PreparationEngine>();
    std::atomic<bool> entered_init{false};
    next->on_initialize = [&] { entered_init = true; };
    auto admitted = runtime.load_engine(std::move(next), {});
    ASSERT_TRUE(admitted.ok());
    ASSERT_TRUE(gate->wait_entered());
    ASSERT_FALSE(entered_init);
    ASSERT_FALSE(runtime.is_loaded());
    ASSERT_TRUE(runtime.cancel_load(admitted.load_id));
    gate->release();
    auto result = wait_load_terminal(runtime, std::move(admitted));
    ASSERT_EQ(result.error, ChorusError::Cancelled);
    ASSERT_FALSE(entered_init);
}

TEST(RuntimePreparation, Reload_joins_gated_preparation_before_old_destruction_and_replacement_initialization) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    service->gate_at = 1;
    std::atomic<bool> old_destroyed{false};
    engine->on_shutdown = [&] { EXPECT_TRUE(service->gate_exited); };
    engine->on_destroy = [&] { old_destroyed = true; };
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    auto old = runtime.render_prompt(chat());
    ASSERT_TRUE(service->wait_entered());
    ASSERT_TRUE(runtime.cancel(old.request_id));
    auto replacement = std::make_unique<PreparationEngine>();
    replacement->on_initialize = [&] {
        EXPECT_TRUE(old_destroyed);
        EXPECT_TRUE(service->closed);
    };
    std::atomic<bool> replacing{false};
    std::thread unblock([&] {
        while (!replacing)
            std::this_thread::yield();
        service->release();
    });
    replacing = true;
    auto admission = runtime.load_engine(std::move(replacement), {});
    ASSERT_TRUE(admission.ok());
    ASSERT_FALSE(runtime.is_loaded());
    unblock.join();
    auto loaded = wait_load_terminal(runtime, std::move(admission));
    ASSERT_TRUE(loaded.ok());
    auto events = loaded.events;
    std::erase_if(events, [](const auto& event) { return event.kind == RuntimeEvent::Kind::ModelLoadProgress; });
    ASSERT_EQ(events.size(), 2U);
    ASSERT_EQ(events[1].kind, RuntimeEvent::Kind::ModelLoaded);
    ASSERT_EQ(events[0].request_id, old.request_id);
    ASSERT_EQ(events[0].error, ChorusError::Cancelled);
    auto fresh = runtime.render_prompt(chat());
    ASSERT_TRUE(fresh.ok());
    ASSERT_EQ(drain_runtime_events(runtime)[0].request_id, fresh.request_id);
}

TEST(RuntimePreparation, Buffered_preparation_success_and_cancelled_chat_drain_after_engine_teardown) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    service->gate_at = 2;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ReleaseGate release{service};
    auto counted = runtime.count_message_tokens(MessageContent::text("draft"));
    auto interrupted = runtime.submit(chat());
    ASSERT_TRUE(counted.ok() && interrupted.ok());
    ASSERT_TRUE(service->wait_entered());
    service->release();
    runtime.stop_all();
    auto loaded = wait_load_terminal(runtime, runtime.load_engine(std::make_unique<PreparationEngine>(), {}));
    ASSERT_TRUE(loaded.ok());
    auto events = loaded.events;
    std::erase_if(events, [](const auto& event) { return event.kind == RuntimeEvent::Kind::ModelLoadProgress; });
    ASSERT_EQ(events.size(), 3U);
    ASSERT_EQ(events[2].kind, RuntimeEvent::Kind::ModelLoaded);
    ASSERT_EQ(events[0].request_id, counted.request_id);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::MessageTokenCount);
    ASSERT_EQ(events[0].token_count, 5);
    ASSERT_EQ(events[1].request_id, interrupted.request_id);
    ASSERT_EQ(events[1].kind, RuntimeEvent::Kind::Error);
    ASSERT_EQ(events[1].error, ChorusError::Cancelled);
    ASSERT_TRUE(runtime.export_conversation_history("npc").empty());
    ASSERT_EQ(runtime.last_turn_outcome("npc"), TurnOutcome::Cancelled);
    auto fresh = runtime.submit(chat());
    ASSERT_TRUE(fresh.ok());
    ASSERT_EQ(drain_runtime_events(runtime).back().request_id, fresh.request_id);
    ASSERT_TRUE(runtime.poll().empty());
}

TEST(RuntimePreparation, Node_cache_eviction_and_reimport_with_same_durable_ids_recompute_counts) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->mock_per_request_context = 100000;
    auto service = engine->service;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    std::vector<ConversationMessage> messages;
    for (size_t i = 0; i < kMessageCountCacheEntries + 1; ++i)
        messages.push_back({static_cast<MessageId>(i), {MessageRole::System, MessageContent::text("x")}});
    ASSERT_FALSE(runtime.import_conversation_history("npc", messages).has_value());
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    drain_runtime_events(runtime);
    auto cold = service->count_calls;
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls - cold, kMessageCountCacheEntries + 1);
    ASSERT_FALSE(runtime.clear_conversation_history("npc"));
    ASSERT_FALSE(runtime.import_conversation_history("npc", {{0, {MessageRole::System, MessageContent::text("x")}}}));
    cold = service->count_calls;
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, cold + 1);
}

TEST(RuntimePreparation, Regeneration_reuses_durable_id_but_not_the_replaced_content_count) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->tokens = {"new reply"};
    auto service = engine->service;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ASSERT_FALSE(runtime.import_conversation_history(
        "npc",
        {{4, {MessageRole::User, MessageContent::text("question")}},
         {5, {MessageRole::Assistant, MessageContent::text("old reply")}}}
    ));
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    drain_runtime_events(runtime);
    ASSERT_TRUE(runtime.regenerate(chat("npc", "")).ok());
    drain_runtime_events(runtime);
    const auto before = service->count_calls;
    ASSERT_TRUE(runtime.render_prompt(chat()).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, before + 1);
    ASSERT_EQ(runtime.export_conversation_history("npc").back().id, 5);
    ASSERT_EQ(*joined_text(runtime.export_conversation_history("npc").back().message.content), "new reply");
}

TEST(RuntimePreparation, Injections_and_systems_survive_worker_fitting_and_zero_id_is_omitted) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->mock_per_request_context = 15;
    ASSERT_TRUE(load_runtime(runtime, std::move(engine), {}).ok());
    ASSERT_FALSE(runtime.import_conversation_history(
        "npc",
        {{9, {MessageRole::System, MessageContent::text("S")}},
         {0, {MessageRole::User, MessageContent::text("old")}},
         {1, {MessageRole::Assistant, MessageContent::text("reply")}}}
    ));
    auto request = chat();
    request.inject = {
        {{MessageRole::System, MessageContent::text("I")}, 100}, {{MessageRole::User, MessageContent::text("J")}, 0}
    };
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    const auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_EQ(events[0].text, "S|I|new|J|");
    ASSERT_EQ(events[0].omitted_message_ids, (std::vector<MessageId>{0, 1}));
}

} // namespace
