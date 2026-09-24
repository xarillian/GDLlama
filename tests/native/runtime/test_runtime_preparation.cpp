#include "chorus/runtime/runtime.hpp"
#include "chorus/runtime/runtime_preparation.hpp"
#include "support/runtime_test_utils.hpp"
#include "support/sync_mock_engine.hpp"

#include <condition_variable>
#include <functional>
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
    std::function<std::variant<RenderedPrompt, RequestRejection>(const std::vector<ChatMessage>&)> render;
    std::optional<int64_t> count_override;
    std::optional<RequestRejection> count_rejection;

    void hook(Hook kind, const std::string& text = {}) const {
        std::unique_lock<std::mutex> lock(mutex);
        threads.insert(std::this_thread::get_id());
        ++calls;
        if (kind == Hook::Count) { ++count_calls; count_bytes += text.size(); counted.push_back(text); }
        if (kind == Hook::Render) ++renders;
        if (calls == gate_at) {
            entered = true;
            cv.notify_all();
            cv.wait(lock, [&] { return released; });
            gate_exited = true;
            if (throw_kind == 1) throw std::runtime_error("preparation test failure");
            if (throw_kind == 2) throw 42;
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
        if (closed) return RequestRejection{ChorusError::EngineNotReady, "closed"};
        std::lock_guard<std::mutex> lock(mutex);
        validated_configs.push_back(request.gen_config);
        return std::nullopt;
    }
    std::variant<RenderedPrompt, RequestRejection> render_chat_prompt(
        const std::vector<ChatMessage>& messages, const std::string&, bool
    ) const override {
        hook(Hook::Render);
        if (closed) return RequestRejection{ChorusError::EngineNotReady, "closed"};
        if (render) return render(messages);
        RenderedPrompt value;
        for (const auto& message : messages) {
            value.text += *joined_text(message.content) + "|";
            value.token_count += static_cast<int32_t>(joined_text(message.content)->size()) + 2;
        }
        return value;
    }
    std::variant<int64_t, RequestRejection> count_message_tokens(const std::string& text) const override {
        hook(Hook::Count, text);
        if (closed) return RequestRejection{ChorusError::EngineNotReady, "closed"};
        if (count_rejection) return *count_rejection;
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
    PreparationEngine() { supports_render = true; mock_per_request_context = 64; }
    ~PreparationEngine() override { if (on_destroy) on_destroy(); }
    EngineCapabilities capabilities() const override {
        auto value = SyncMockEngine::capabilities();
        value.message_token_counting = counting;
        return value;
    }
    std::optional<InitializationFailure> initialize(const ChorusConfig& config, Logger logger) override {
        if (on_initialize) on_initialize();
        return SyncMockEngine::initialize(config, std::move(logger));
    }
    std::shared_ptr<RequestPreparation> request_preparation() const override { return service; }
    void shutdown() override {
        if (on_shutdown) on_shutdown();
        service->closed = true;
        SyncMockEngine::shutdown();
    }
};

struct ReleaseGate {
    std::shared_ptr<GatedPreparation> service;
    ~ReleaseGate() { service->release(); }
};

GenerationRequest chat(const char* session = "npc", std::string prompt = "new") {
    GenerationRequest request;
    request.session_id = session;
    request.prompt = std::move(prompt);
    request.overrides.max_tokens = ConfigPatch<int32_t>::set(0);
    return request;
}
GenerationRequest raw(std::string text) {
    GenerationRequest request;
    request.prompt = std::move(text);
    return request;
}

TEST(RuntimePreparation, Gated_hook_leaves_every_admission_poll_and_cancel_responsive_with_one_worker_and_bounded_queue) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ReleaseGate release{service};
    ASSERT_FALSE(runtime.import_conversation_history("reroll", {
        {10, {MessageRole::User, MessageContent::text("question")}},
        {11, {MessageRole::Assistant, MessageContent::text("reply")}}
    }).has_value());
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
        ASSERT_EQ(service->threads.size(), 1U);
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

TEST(RuntimePreparation, Preview_freezes_history_defaults_and_source_without_occupying_or_creating_history) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    service->gate_at = 1;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ReleaseGate release{service};
    ASSERT_FALSE(runtime.import_conversation_history("npc", {{0, {MessageRole::System, MessageContent::text("old")}}}).has_value());
    auto request = chat();
    auto preview = runtime.render_prompt(request);
    ASSERT_TRUE(service->wait_entered());
    ASSERT_FALSE(runtime.active_request_for_session("npc"));
    ASSERT_FALSE(runtime.edit_message("npc", 0, MessageContent::text("edited")).has_value());
    request.prompt = "changed";
    HostDefaults defaults;
    defaults.config.max_tokens = ConfigPatch<int32_t>::set(INT32_MAX);
    runtime.set_host_defaults(defaults);
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {
        {0, {MessageRole::User, MessageContent::text("oversized-old-content")}},
        {1, {MessageRole::Assistant, MessageContent::text("reply")}}
    }).has_value());
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {
        {0, {MessageRole::System, MessageContent::text("S")}},
        {1, {MessageRole::System, MessageContent::text("T")}}
    }));
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_TRUE(runtime.submit(chat()).ok());
    auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].error, ChorusError::Tokenize);
    ASSERT_TRUE(observed->submitted_ids.empty());
    ASSERT_TRUE(runtime.export_conversation_history("npc").empty());
    auto replacement = std::make_unique<PreparationEngine>();
    replacement->service->count_rejection = RequestRejection{ChorusError::Tokenize, "raw tokenization failed"};
    ASSERT_FALSE(runtime.load_engine(std::move(replacement), {}).has_value());
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text("text")).ok());
    events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::Error);
    ASSERT_EQ(events[0].error, ChorusError::Tokenize);
}

TEST(RuntimePreparation, Counts_join_parts_and_bound_arbitrary_content_cache_and_reset_on_reload) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
        ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text(std::string(kContentCountCacheBytes / 2, fill))).ok());
        drain_runtime_events(runtime);
    }
    before = service->count_calls;
    ASSERT_TRUE(runtime.count_message_tokens(MessageContent::text(std::string(kContentCountCacheBytes / 2, 'a'))).ok());
    drain_runtime_events(runtime);
    ASSERT_EQ(service->count_calls, before + 1);
    auto replacement = std::make_unique<PreparationEngine>();
    auto fresh = replacement->service;
    ASSERT_FALSE(runtime.load_engine(std::move(replacement), {}).has_value());
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
        if (event.kind == RuntimeEvent::Kind::EngineFailed) { ++failures; continue; }
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
        if (event.kind == RuntimeEvent::Kind::EngineFailed) continue;
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
    auto engine = std::make_unique<PreparationEngine>();
    auto* observed = engine.get();
    auto service = engine->service;
    service->gate_at = 3;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
        ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
        ReleaseGate release{service};
        auto accepted = count ? runtime.count_message_tokens(MessageContent::text("draft")) : runtime.render_prompt(chat());
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ReleaseGate release{service};
    HostDefaults defaults;
    defaults.config.temperature = ConfigPatch<float>::set(0.25f);
    runtime.set_host_defaults(defaults);
    auto results = runtime.submit_batch(std::vector<GenerationRequest>{raw("first"), raw("second"), raw("third")});
    ASSERT_TRUE(service->wait_entered());
    defaults.config.temperature = ConfigPatch<float>::set(0.75f);
    runtime.set_host_defaults(defaults);
    service->release();
    auto events = drain_runtime_events(runtime, 3);
    ASSERT_EQ(events.size(), 3U);
    ASSERT_EQ(service->validated_configs.size(), 3U);
    for (const auto& config : service->validated_configs)
        ASSERT_EQ(config.temperature, 0.25f);
}

TEST(RuntimePreparation, Reload_joins_gated_preparation_before_old_destruction_and_replacement_initialization) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    auto service = engine->service;
    service->gate_at = 1;
    bool old_destroyed = false;
    engine->on_shutdown = [&] { EXPECT_TRUE(service->gate_exited); };
    engine->on_destroy = [&] { old_destroyed = true; };
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ReleaseGate release{service};
    auto old = runtime.render_prompt(chat());
    ASSERT_TRUE(service->wait_entered());
    ASSERT_TRUE(runtime.cancel(old.request_id));
    auto replacement = std::make_unique<PreparationEngine>();
    replacement->on_initialize = [&] { EXPECT_TRUE(old_destroyed); EXPECT_TRUE(service->closed); };
    std::atomic<bool> replacing{false};
    std::thread unblock([&] {
        while (!replacing) std::this_thread::yield();
        service->release();
    });
    replacing = true;
    auto error = runtime.load_engine(std::move(replacement), {});
    unblock.join();
    ASSERT_FALSE(error);
    auto events = runtime.poll();
    ASSERT_EQ(events.size(), 1U);
    ASSERT_EQ(events[0].request_id, old.request_id);
    ASSERT_EQ(events[0].error, ChorusError::Cancelled);
    auto fresh = runtime.render_prompt(chat());
    ASSERT_TRUE(fresh.ok());
    ASSERT_EQ(drain_runtime_events(runtime)[0].request_id, fresh.request_id);
}

TEST(RuntimePreparation, Node_cache_eviction_and_reimport_with_same_durable_ids_recompute_counts) {
    ChorusRuntime runtime;
    auto engine = std::make_unique<PreparationEngine>();
    engine->mock_per_request_context = 100000;
    auto service = engine->service;
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {
        {4, {MessageRole::User, MessageContent::text("question")}},
        {5, {MessageRole::Assistant, MessageContent::text("old reply")}}
    }));
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
    ASSERT_FALSE(runtime.load_engine(std::move(engine), {}).has_value());
    ASSERT_FALSE(runtime.import_conversation_history("npc", {
        {9, {MessageRole::System, MessageContent::text("S")}},
        {0, {MessageRole::User, MessageContent::text("old")}},
        {1, {MessageRole::Assistant, MessageContent::text("reply")}}
    }));
    auto request = chat();
    request.inject = {{{MessageRole::System, MessageContent::text("I")}, 100},
                      {{MessageRole::User, MessageContent::text("J")}, 0}};
    ASSERT_TRUE(runtime.render_prompt(request).ok());
    const auto events = drain_runtime_events(runtime);
    ASSERT_EQ(events[0].kind, RuntimeEvent::Kind::PromptRendered);
    ASSERT_EQ(events[0].text, "S|I|new|J|");
    ASSERT_EQ(events[0].omitted_message_ids, (std::vector<MessageId>{0, 1}));
}

} // namespace
