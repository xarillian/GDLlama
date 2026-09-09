#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/providers/llama/llama_batch_planner.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"
#include "chorus/providers/llama/llama_recovery_planner.hpp"
#include "chorus/providers/llama/llama_scheduler.hpp"
#include "chorus/providers/llama/llama_sequence_id_pool.hpp"
#include "gtest_utils.hpp"

class LlamaSchedulerModelTest : public ChorusModelTest {};

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct SchedulerObservation {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::LlamaBatchRecord> batches;
    std::map<Chorus::RequestId, std::vector<Chorus::ChorusSignal>> terminals;
    std::vector<Chorus::RequestId> terminal_order;
    Chorus::RequestId gate_request_id = -1;
    bool gate_seen = false;
    bool release_gate = false;

    void observe(const Chorus::LlamaBatchRecord& record) {
        std::unique_lock<std::mutex> lock(mutex);
        batches.push_back(record);
        const auto observed_gate = gate_request_id;
        if (std::find(record.request_ids.begin(), record.request_ids.end(), observed_gate) == record.request_ids.end())
            return;
        gate_seen = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_gate; });
        if (gate_request_id == observed_gate)
            gate_request_id = -1;
        release_gate = false;
        cv.notify_all();
    }

    void handle(const Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::lock_guard<std::mutex> lock(mutex);
        terminals[signal.request_id].push_back(signal);
        terminal_order.push_back(signal.request_id);
        cv.notify_all();
    }

    bool wait_for_gate() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(15), [&] { return gate_seen; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        release_gate = true;
        cv.notify_all();
    }

    bool wait_for_terminals(const std::vector<Chorus::RequestId>& ids) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(30), [&] {
            return std::ranges::all_of(ids, [&](Chorus::RequestId id) { return !terminals[id].empty(); });
        });
    }
};

Chorus::ChorusRequest make_scheduler_generation(Chorus::RequestId id, int priority) {
    Chorus::ChorusRequest request;
    request.id = id;
    request.priority = priority;
    request.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    request.gen_config.max_tokens = 1;
    return request;
}

Chorus::ChorusRequest make_scheduler_embedding(Chorus::RequestId id, std::string prompt, int priority) {
    Chorus::ChorusRequest request;
    request.id = id;
    request.type = Chorus::RequestType::Embedding;
    request.priority = priority;
    request.prompt = std::move(prompt);
    return request;
}

bool record_contains(const Chorus::LlamaBatchRecord& record, Chorus::RequestId id) {
    return std::find(record.request_ids.begin(), record.request_ids.end(), id) != record.request_ids.end();
}

} // namespace

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

static Chorus::ChorusConfig make_gguf_config(const std::string& path) {
    Chorus::ChorusConfig config;
    config.model.model_id = "test-model";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, path});
    return config;
}

TEST_F(LlamaSchedulerModelTest, Malformed_sampler_requests_terminate_and_engine_remains_usable) {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    SchedulerObservation state;
    Chorus::LlamaEngine engine;
    ASSERT_FALSE(engine.initialize(config, {}).has_value());

    auto malformed = make_scheduler_generation(1, 0);
    malformed.on_event = [&](const auto& signal) { state.handle(signal); };
    malformed.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"repeat_penalty", 0.0}};
    auto rejection = engine.validate_request(malformed);
    ASSERT_TRUE(rejection);
    EXPECT_EQ(rejection->error, Chorus::ChorusError::UnsupportedOption);
    engine.submit_request(malformed);
    ASSERT_TRUE(state.wait_for_terminals({1}));
    EXPECT_TRUE(engine.is_initialized());

    auto grammar = make_scheduler_generation(2, 0);
    grammar.on_event = malformed.on_event;
    grammar.gen_config.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "not a grammar"};
    ASSERT_FALSE(engine.validate_request(grammar));
    engine.submit_request(grammar);
    ASSERT_TRUE(state.wait_for_terminals({2}));
    EXPECT_TRUE(engine.is_initialized());

    auto valid = make_scheduler_generation(3, 0);
    valid.on_event = malformed.on_event;
    engine.submit_request(valid);
    ASSERT_TRUE(state.wait_for_terminals({3}));
    engine.shutdown();
    ASSERT_EQ(state.terminals[1].size(), size_t{1});
    EXPECT_EQ(std::get<Chorus::ChorusSignal::Error>(state.terminals[1][0].event).code,
              Chorus::ChorusError::UnsupportedOption);
    ASSERT_EQ(state.terminals[2].size(), size_t{1});
    EXPECT_EQ(std::get<Chorus::ChorusSignal::Error>(state.terminals[2][0].event).code,
              Chorus::ChorusError::InvalidRequest);
    ASSERT_EQ(state.terminals[3].size(), size_t{1});
    EXPECT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state.terminals[3][0].event));
}

TEST_F(LlamaSchedulerModelTest, Unexpected_batch_exception_fences_admission_and_drains_active_and_queued) {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    SchedulerObservation state;
    state.gate_request_id = 1;
    LlamaScheduler scheduler;
    ASSERT_FALSE(scheduler.initialize(config, {}).has_value());
    scheduler.set_batch_observer([&](const auto& record) {
        state.observe(record);
        throw std::logic_error("injected batch failure");
    });
    auto active = make_scheduler_generation(1, 10);
    active.on_event = [&](const auto& signal) { state.handle(signal); };
    ASSERT_TRUE(scheduler.push_request(active));
    const bool gated = state.wait_for_gate();
    if (!gated) {
        state.release();
        FAIL() << "Worker did not reach batch gate";
    }
    auto queued = make_scheduler_generation(2, 0);
    queued.on_event = active.on_event;
    EXPECT_TRUE(scheduler.push_request(queued));
    auto throwing_sink = make_scheduler_generation(3, 20);
    throwing_sink.on_event = [&](const auto& signal) {
        state.handle(signal);
        throw std::runtime_error("contract-violating sink");
    };
    EXPECT_TRUE(scheduler.push_request(throwing_sink));
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({1, 2, 3}));
    EXPECT_FALSE(scheduler.is_healthy());
    EXPECT_FALSE(scheduler.push_request(make_scheduler_generation(4, 0)));
    scheduler.shutdown();
    EXPECT_EQ(state.batches.size(), size_t{1});
    for (int id : {1, 2, 3}) {
        ASSERT_EQ(state.terminals[id].size(), size_t{1});
        EXPECT_EQ(std::get<Chorus::ChorusSignal::Error>(state.terminals[id][0].event).code,
                  Chorus::ChorusError::Unknown);
    }
}

TEST_F(LlamaSchedulerModelTest, Unexpected_admission_exception_retains_preparing_and_buffered_terminal_ownership) {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    SchedulerObservation state;
    state.gate_request_id = 1;
    LlamaScheduler scheduler;
    ASSERT_FALSE(scheduler.initialize(config, {}).has_value());
    scheduler.set_batch_observer([&](const auto& record) { state.observe(record); });
    scheduler.set_admission_observer([](Chorus::RequestId id) {
        if (id == 3)
            throw 42;
    });
    auto blocker = make_scheduler_generation(1, 0);
    blocker.on_event = [&](const auto& signal) { state.handle(signal); };
    ASSERT_TRUE(scheduler.push_request(blocker));
    const bool gated = state.wait_for_gate();
    if (!gated) {
        state.release();
        FAIL() << "Worker did not reach batch gate";
    }
    auto zero = make_scheduler_generation(2, 0);
    zero.gen_config.max_tokens = 0;
    zero.on_event = blocker.on_event;
    auto preparing = make_scheduler_generation(3, 0);
    preparing.on_event = blocker.on_event;
    auto queued = make_scheduler_generation(4, 0);
    queued.on_event = blocker.on_event;
    EXPECT_TRUE(scheduler.push_request(zero));
    EXPECT_TRUE(scheduler.push_request(preparing));
    EXPECT_TRUE(scheduler.push_request(queued));
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({1, 2, 3, 4}));
    EXPECT_FALSE(scheduler.is_healthy());
    scheduler.shutdown();
    ASSERT_EQ(state.terminals[1].size(), size_t{1});
    EXPECT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state.terminals[1][0].event));
    for (int id : {2, 3, 4}) {
        ASSERT_EQ(state.terminals[id].size(), size_t{1});
        EXPECT_EQ(std::get<Chorus::ChorusSignal::Error>(state.terminals[id][0].event).code,
                  Chorus::ChorusError::Unknown);
    }
}

TEST_F(LlamaSchedulerModelTest, Preparing_cancellation_is_consumed_and_late_or_unknown_ids_leave_worker_idle) {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    SchedulerObservation state;
    state.gate_request_id = 1;
    LlamaScheduler scheduler;
    ASSERT_FALSE(scheduler.initialize(config, {}).has_value());
    scheduler.set_admission_observer([&](Chorus::RequestId id) {
        state.observe({Chorus::RequestType::Generate, 0, {id}, {}});
    });
    auto preparing = make_scheduler_generation(1, 0);
    preparing.on_event = [&](const auto& signal) { state.handle(signal); };
    ASSERT_TRUE(scheduler.push_request(preparing));
    const bool gated = state.wait_for_gate();
    if (!gated) {
        state.release();
        FAIL() << "Worker did not reach admission gate";
    }
    scheduler.cancel_request(preparing.id);
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({1}));
    auto zero = make_scheduler_generation(2, 0);
    zero.gen_config.max_tokens = 0;
    zero.on_event = [&](const auto& signal) {
        state.handle(signal);
        scheduler.cancel_request(signal.request_id);
    };
    ASSERT_TRUE(scheduler.push_request(zero));
    ASSERT_TRUE(state.wait_for_terminals({2}));
    scheduler.cancel_request(preparing.id);
    scheduler.cancel_request(zero.id);
    scheduler.cancel_request(9999);
    const uint64_t before = scheduler.worker_iterations();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_LT(scheduler.worker_iterations() - before, uint64_t{50});
    EXPECT_TRUE(scheduler.is_healthy());
    auto valid = make_scheduler_generation(3, 0);
    valid.on_event = preparing.on_event;
    ASSERT_TRUE(scheduler.push_request(valid));
    ASSERT_TRUE(state.wait_for_terminals({3}));
    scheduler.shutdown();
    ASSERT_EQ(state.terminals[1].size(), size_t{1});
    EXPECT_EQ(std::get<Chorus::ChorusSignal::Error>(state.terminals[1][0].event).code,
              Chorus::ChorusError::Cancelled);
    for (int id : {2, 3}) {
        ASSERT_EQ(state.terminals[id].size(), size_t{1});
        EXPECT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state.terminals[id][0].event));
    }
}

TEST(LlamaScheduler, sequence_ids_allocate_reuse_and_reject_invalid_release) {
    Chorus::LlamaSequenceIdPool pool(2);
    ASSERT_EQ(pool.acquire(), std::optional<int>{0});
    ASSERT_EQ(pool.acquire(), std::optional<int>{1});
    ASSERT_EQ(pool.acquire(), std::nullopt);
    pool.release(0);
    ASSERT_EQ(pool.acquire(), std::optional<int>{0});
    EXPECT_THROW(pool.release(4), std::logic_error);
    pool.release(0);
    EXPECT_THROW(pool.release(0), std::logic_error);
}

TEST(LlamaScheduler, pure_generation_planner_reserves_decode_tokens_before_prefill) {
    const std::vector<Chorus::LlamaPlannerSequence> sequences{
        {1, Chorus::RequestType::Generate, 4, 10, Chorus::LlamaPlannerPhase::Decode, 0, 0, false},
        {2, Chorus::RequestType::Generate, 4, 20, Chorus::LlamaPlannerPhase::Decode, 0, 0, false},
        {3, Chorus::RequestType::Generate, 4, 30, Chorus::LlamaPlannerPhase::Prefill, 0, 6, false},
    };
    const auto plan = Chorus::llama_plan_batch(sequences, 3, 8, 0, std::nullopt);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->entries.size(), size_t{3});
    EXPECT_EQ(plan->entries[0].sequence_id, 1);
    EXPECT_EQ(plan->entries[1].sequence_id, 2);
    EXPECT_EQ(plan->entries[2].sequence_id, 3);
    EXPECT_TRUE(plan->entries[0].decode);
    EXPECT_TRUE(plan->entries[1].decode);
    EXPECT_FALSE(plan->entries[2].decode);
    EXPECT_EQ(sequences[2].prompt_cursor, size_t{0});
}

TEST(LlamaScheduler, pure_generation_planner_rotates_equal_priority_decoders) {
    const std::vector<Chorus::LlamaPlannerSequence> sequences{
        {1, Chorus::RequestType::Generate, 4, 10, Chorus::LlamaPlannerPhase::Decode, 0, 0, false},
        {2, Chorus::RequestType::Generate, 4, 20, Chorus::LlamaPlannerPhase::Decode, 0, 0, false},
    };
    const auto first = Chorus::llama_plan_batch(sequences, 1, 8, 0, std::nullopt);
    ASSERT_TRUE(first);
    ASSERT_TRUE(first->decode_fairness);
    const auto second = Chorus::llama_plan_batch(sequences, 1, 8, *first->decode_fairness, std::nullopt);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->entries[0].sequence_id, 1);
    EXPECT_EQ(second->entries[0].sequence_id, 2);
}

TEST(LlamaScheduler, pure_homogeneous_planner_alternates_and_never_splits_embeddings) {
    const std::vector<Chorus::LlamaPlannerSequence> sequences{
        {1, Chorus::RequestType::Embedding, 4, 10, Chorus::LlamaPlannerPhase::Prefill, 0, 3, false},
        {2, Chorus::RequestType::Generate, 4, 20, Chorus::LlamaPlannerPhase::Prefill, 0, 4, false},
        {3, Chorus::RequestType::Embedding, 4, 30, Chorus::LlamaPlannerPhase::Prefill, 0, 4, false},
    };
    const auto first = Chorus::llama_plan_batch(sequences, 8, 6, 0, std::nullopt);
    ASSERT_TRUE(first);
    EXPECT_EQ(first->type, Chorus::RequestType::Embedding);
    EXPECT_EQ(first->participants, std::vector<int>({1}));
    const auto second = Chorus::llama_plan_batch(sequences, 8, 6, 0, first->contested_type);
    ASSERT_TRUE(second);
    EXPECT_EQ(second->type, Chorus::RequestType::Generate);
    EXPECT_TRUE(std::ranges::all_of(second->entries, [](const auto& entry) { return entry.sequence_id == 2; }));
}

TEST(LlamaScheduler, scripted_recovery_reduces_only_failed_contributors_before_commit) {
    const std::vector<Chorus::LlamaRecoveryContribution> contributors{{1, 4, 1}, {2, 3, 1}, {3, 1, 1}};
    const auto retries = Chorus::llama_recovery_reductions(contributors);
    ASSERT_EQ(retries.size(), size_t{7});
    EXPECT_EQ(retries.front().entry_limits, std::vector<size_t>({4, 2, 1}));
    EXPECT_EQ(retries[4].entry_limits, std::vector<size_t>({1, 1, 1}));
    EXPECT_EQ(retries[5].sequence_ids, std::vector<int>({1, 2}));
    EXPECT_EQ(retries[6].sequence_ids, std::vector<int>({1}));

    std::map<int, size_t> committed{{1, 0}, {2, 0}, {3, 0}, {99, 0}};
    for (const auto& retry : retries) {
        EXPECT_EQ(std::ranges::find(retry.sequence_ids, 99), retry.sequence_ids.end());
        if (retry.sequence_ids != std::vector<int>({1, 2}))
            continue;
        for (size_t index = 0; index < retry.sequence_ids.size(); ++index)
            committed[retry.sequence_ids[index]] += retry.entry_limits[index];
        break;
    }
    EXPECT_EQ(committed[1], size_t{1});
    EXPECT_EQ(committed[2], size_t{1});
    EXPECT_EQ(committed[3], size_t{0});
    EXPECT_EQ(committed[99], size_t{0});
}

TEST(LlamaScheduler, load_option_CPU_placement_disables_every_offload_path) {
    Chorus::LlamaLoadConfig load;
    load.use_gpu = false;
    Chorus::LlamaOffloadDeviceList no_offload_devices{};

    const auto model = Chorus::make_llama_model_params(load, no_offload_devices);
    const auto context = Chorus::make_llama_context_params(load);

    ASSERT_EQ(model.n_gpu_layers, int32_t{0});
    ASSERT_TRUE(model.devices == no_offload_devices.data());
    ASSERT_TRUE(model.devices[0] == nullptr);
    ASSERT_TRUE(!context.offload_kqv);
    ASSERT_TRUE(!context.op_offload);
}

TEST(LlamaScheduler, load_option_GPU_placement_preserves_upstream_device_selection) {
    Chorus::LlamaLoadConfig load;
    load.use_gpu = true;
    load.gpu_layers = 17;
    load.main_gpu = 2;
    Chorus::LlamaOffloadDeviceList no_offload_devices{};

    const auto model = Chorus::make_llama_model_params(load, no_offload_devices);
    const auto context = Chorus::make_llama_context_params(load);

    ASSERT_EQ(model.n_gpu_layers, int32_t{17});
    ASSERT_EQ(model.main_gpu, int32_t{2});
    ASSERT_TRUE(model.devices == nullptr);
    ASSERT_TRUE(context.offload_kqv);
    ASSERT_TRUE(context.op_offload);
}

TEST(LlamaScheduler, embeddings_require_a_single_supported_architecture) {
    ASSERT_TRUE(Chorus::llama_embedding_architecture_supported(true, false));
    ASSERT_TRUE(Chorus::llama_embedding_architecture_supported(false, true));
    ASSERT_TRUE(!Chorus::llama_embedding_architecture_supported(false, false));
    ASSERT_TRUE(!Chorus::llama_embedding_architecture_supported(true, true));
}

TEST(LlamaScheduler, load_option_context_params_forward_exact_values) {
    Chorus::LlamaLoadConfig load;
    load.context_size = 4096;
    load.max_concurrent_requests = 3;
    load.thread_count = 6;
    load.n_batch = 96;
    load.n_ubatch = 32;
    load.pooling = LLAMA_POOLING_TYPE_LAST;
    const auto params = Chorus::make_llama_context_params(load);
    ASSERT_EQ(params.n_ctx, uint32_t{4096});
    ASSERT_EQ(params.n_seq_max, uint32_t{3});
    ASSERT_EQ(params.n_threads, int32_t{6});
    ASSERT_EQ(params.n_threads_batch, int32_t{6});
    ASSERT_EQ(params.n_batch, uint32_t{96});
    ASSERT_EQ(params.n_ubatch, uint32_t{32});
    ASSERT_EQ(params.pooling_type, LLAMA_POOLING_TYPE_LAST);
}

TEST_F(LlamaSchedulerModelTest, Mixed_requests_follow_priority_fifo_and_use_homogeneous_batches) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{3}},
        {"n_batch", int64_t{64}},
        {"n_ubatch", int64_t{16}},
        {"pooling", std::string{"none"}},
    };

    SchedulerObservation state;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}).has_value());
    engine.set_batch_observer([&](const Chorus::LlamaBatchRecord& record) { state.observe(record); });

    auto blocker = make_scheduler_generation(1, 100);
    blocker.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.gate_request_id = blocker.id;
    }
    engine.submit_request(blocker);
    ASSERT_TRUE(state.wait_for_gate());

    auto high_embedding = make_scheduler_embedding(2, "short embedding prompt", 10);
    high_embedding.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    auto low_generation = make_scheduler_generation(3, 0);
    low_generation.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    engine.submit_request(high_embedding);
    engine.submit_request(low_generation);
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({blocker.id, high_embedding.id, low_generation.id}));

    auto fifo_blocker = make_scheduler_generation(4, 100);
    fifo_blocker.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.gate_request_id = fifo_blocker.id;
        state.gate_seen = false;
    }
    engine.submit_request(fifo_blocker);
    ASSERT_TRUE(state.wait_for_gate());

    auto first_generation = make_scheduler_generation(5, 5);
    first_generation.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    auto second_embedding = make_scheduler_embedding(6, "another short embedding prompt", 5);
    second_embedding.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    engine.submit_request(first_generation);
    engine.submit_request(second_embedding);
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({fifo_blocker.id, first_generation.id, second_embedding.id}));
    engine.shutdown();

    std::lock_guard<std::mutex> lock(state.mutex);
    const auto high = std::find_if(state.batches.begin(), state.batches.end(), [&](const auto& record) {
        return record_contains(record, high_embedding.id) || record_contains(record, low_generation.id);
    });
    ASSERT_TRUE(high != state.batches.end());
    ASSERT_TRUE(high->type == Chorus::RequestType::Embedding);
    ASSERT_TRUE(record_contains(*high, high_embedding.id));
    ASSERT_EQ(high->embedding_output_indices.size(), size_t{1});
    ASSERT_EQ(high->embedding_output_indices[0], high->token_count - 1);

    const auto fifo = std::find_if(state.batches.begin(), state.batches.end(), [&](const auto& record) {
        return record_contains(record, first_generation.id) || record_contains(record, second_embedding.id);
    });
    ASSERT_TRUE(fifo != state.batches.end());
    ASSERT_TRUE(fifo->type == Chorus::RequestType::Generate);
    ASSERT_TRUE(record_contains(*fifo, first_generation.id));

    const std::map<Chorus::RequestId, Chorus::RequestType> types{
        {blocker.id, Chorus::RequestType::Generate},
        {high_embedding.id, Chorus::RequestType::Embedding},
        {low_generation.id, Chorus::RequestType::Generate},
        {fifo_blocker.id, Chorus::RequestType::Generate},
        {first_generation.id, Chorus::RequestType::Generate},
        {second_embedding.id, Chorus::RequestType::Embedding},
    };
    for (const auto& record : state.batches)
        for (const auto id : record.request_ids)
            ASSERT_TRUE(types.at(id) == record.type);
}

TEST_F(LlamaSchedulerModelTest, Embedding_batches_respect_n_ubatch_and_cancellation) {
    Chorus::ChorusConfig config = make_gguf_config("tests/models/embeddinggemma-300M-Q8_0.gguf");
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{4}},
        {"n_batch", int64_t{16}},
        {"n_ubatch", int64_t{8}},
        {"pooling", std::string{"none"}},
    };

    SchedulerObservation state;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}).has_value());
    engine.set_batch_observer([&](const Chorus::LlamaBatchRecord& record) { state.observe(record); });

    auto blocker = make_scheduler_embedding(10, "block", 0);
    blocker.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.gate_request_id = blocker.id;
    }
    engine.submit_request(blocker);
    ASSERT_TRUE(state.wait_for_gate());

    std::vector<Chorus::ChorusRequest> embeddings;
    for (const auto [id, prompt] : std::vector<std::pair<Chorus::RequestId, std::string>>{
             {11, "a"}, {12, "b"}, {13, "c"}}) {
        auto request = make_scheduler_embedding(id, prompt, 0);
        request.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
        embeddings.push_back(std::move(request));
    }
    for (const auto& request : embeddings)
        engine.submit_request(request);
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({blocker.id, 11, 12, 13}));
    engine.shutdown();

    std::lock_guard<std::mutex> lock(state.mutex);
    bool batched_embeddings = false;
    for (const auto& record : state.batches) {
        ASSERT_TRUE(record.token_count <= 8);
        if (record.type == Chorus::RequestType::Embedding && record.request_ids.size() > 1)
            batched_embeddings = true;
    }
    ASSERT_TRUE(batched_embeddings);
}

TEST_F(LlamaSchedulerModelTest, Embeddings_cancel_while_queued_and_admitted) {
    auto make_config = [] {
        Chorus::ChorusConfig config = make_gguf_config("tests/models/embeddinggemma-300M-Q8_0.gguf");
        config.provider_options["llama"] = Chorus::ProviderOptionMap{
            {"use_gpu", false},
            {"max_concurrent_requests", int64_t{1}},
            {"n_batch", int64_t{16}},
            {"n_ubatch", int64_t{8}},
            {"pooling", std::string{"none"}},
        };
        return config;
    };

    SchedulerObservation queued_state;
    Chorus::LlamaEngine queued_engine;
    ASSERT_TRUE(!queued_engine.initialize(make_config(), {}).has_value());
    queued_engine.set_batch_observer([&](const Chorus::LlamaBatchRecord& record) { queued_state.observe(record); });
    auto blocker = make_scheduler_embedding(20, "block", 0);
    blocker.on_event = [&](const Chorus::ChorusSignal& signal) { queued_state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(queued_state.mutex);
        queued_state.gate_request_id = blocker.id;
    }
    queued_engine.submit_request(blocker);
    ASSERT_TRUE(queued_state.wait_for_gate());
    auto queued = make_scheduler_embedding(21, "queued", 0);
    queued.on_event = [&](const Chorus::ChorusSignal& signal) { queued_state.handle(signal); };
    queued_engine.submit_request(queued);
    queued_engine.cancel_request(queued.id);
    queued_state.release();
    ASSERT_TRUE(queued_state.wait_for_terminals({blocker.id, queued.id}));
    queued_engine.shutdown();

    {
        std::lock_guard<std::mutex> lock(queued_state.mutex);
        ASSERT_EQ(queued_state.terminals[queued.id].size(), size_t{1});
        const auto* error = std::get_if<Chorus::ChorusSignal::Error>(&queued_state.terminals[queued.id][0].event);
        ASSERT_TRUE(error != nullptr && error->code == Chorus::ChorusError::Cancelled);
        for (const auto& record : queued_state.batches)
            ASSERT_TRUE(!record_contains(record, queued.id));
    }

    SchedulerObservation admitted_state;
    Chorus::LlamaEngine admitted_engine;
    ASSERT_TRUE(!admitted_engine.initialize(make_config(), {}).has_value());
    admitted_engine.set_batch_observer([&](const Chorus::LlamaBatchRecord& record) { admitted_state.observe(record); });
    auto admitted = make_scheduler_embedding(22, "admitted", 0);
    admitted.on_event = [&](const Chorus::ChorusSignal& signal) { admitted_state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(admitted_state.mutex);
        admitted_state.gate_request_id = admitted.id;
    }
    admitted_engine.submit_request(admitted);
    ASSERT_TRUE(admitted_state.wait_for_gate());
    admitted_engine.cancel_request(admitted.id);
    admitted_state.release();
    ASSERT_TRUE(admitted_state.wait_for_terminals({admitted.id}));
    admitted_engine.shutdown();

    std::lock_guard<std::mutex> lock(admitted_state.mutex);
    ASSERT_EQ(admitted_state.terminals[admitted.id].size(), size_t{1});
    const auto* error = std::get_if<Chorus::ChorusSignal::Error>(&admitted_state.terminals[admitted.id][0].event);
    ASSERT_TRUE(error != nullptr && error->code == Chorus::ChorusError::Cancelled);
    ASSERT_TRUE(std::ranges::any_of(admitted_state.batches, [&](const auto& record) {
        return record_contains(record, admitted.id);
    }));
}

TEST_F(LlamaSchedulerModelTest, Active_cancellations_emit_priority_ordered_terminals) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{3}},
    };

    SchedulerObservation state;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}).has_value());
    engine.set_batch_observer([&](const Chorus::LlamaBatchRecord& record) { state.observe(record); });

    auto blocker = make_scheduler_generation(100, 100);
    blocker.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.gate_request_id = blocker.id;
    }
    engine.submit_request(blocker);
    ASSERT_TRUE(state.wait_for_gate());

    auto low = make_scheduler_generation(101, 0);
    low.gen_config.max_tokens = 8;
    low.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    low.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    auto high = make_scheduler_generation(102, 10);
    high.gen_config.max_tokens = 8;
    high.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    high.on_event = [&](const Chorus::ChorusSignal& signal) { state.handle(signal); };
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.gate_request_id = high.id;
        state.gate_seen = false;
    }
    engine.submit_request(low);
    engine.submit_request(high);
    state.release();
    ASSERT_TRUE(state.wait_for_gate());

    engine.cancel_request(low.id);
    engine.cancel_request(high.id);
    state.release();
    ASSERT_TRUE(state.wait_for_terminals({low.id, high.id}));
    engine.shutdown();

    std::lock_guard<std::mutex> lock(state.mutex);
    ASSERT_EQ(state.terminals[low.id].size(), size_t{1});
    ASSERT_EQ(state.terminals[high.id].size(), size_t{1});
    ASSERT_GE(state.terminal_order.size(), size_t{2});
    EXPECT_EQ(state.terminal_order[state.terminal_order.size() - 2], high.id);
    EXPECT_EQ(state.terminal_order.back(), low.id);
}

TEST_F(LlamaSchedulerModelTest, Transient_decode_failure_recovers) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{64}}, // tiny unified KV cache
        {"n_batch", int64_t{64}},      // keep decode-failure setup independent from the larger default batch
        {"n_ubatch", int64_t{32}},
        {"n_batch", int64_t{16}},
        {"max_concurrent_requests", int64_t{1}},
    };

    // A prompt guaranteed to exceed 64 KV cells (~2000 tokens).
    std::string huge_prompt;
    for (int i = 0; i < 200; ++i)
        huge_prompt += "The quick brown fox jumps over the lazy dog. ";

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> oversized_terminals;
        std::vector<Chorus::ChorusSignal> recovery_terminals;
        size_t recovery_tokens = 0;
    } state;

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config, {}).has_value());

    Chorus::ChorusRequest big;
    big.id = 1;
    big.prompt = huge_prompt;
    big.gen_config.max_tokens = 8;
    big.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event))
            return;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.oversized_terminals.push_back(sig);
        state.cv.notify_all();
    };
    engine.submit_request(big);

    bool oversized_finished = false;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        oversized_finished = state.cv.wait_for(lock, std::chrono::seconds(15), [&] {
            return !state.oversized_terminals.empty();
        });
    }
    if (!oversized_finished) {
        engine.shutdown();
        ASSERT_TRUE(oversized_finished);
        return;
    }

    Chorus::ChorusRequest small;
    small.id = 2;
    small.prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    small.gen_config.max_tokens = 4;
    small.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    small.on_event = [&](const Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
            ++state.recovery_tokens;
        } else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
                   std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
            state.recovery_terminals.push_back(sig);
            state.cv.notify_all();
        }
    };
    engine.submit_request(small);

    bool recovery_finished = false;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        recovery_finished = state.cv.wait_for(lock, std::chrono::seconds(15), [&] {
            return !state.recovery_terminals.empty();
        });
    }

    engine.shutdown();

    ASSERT_TRUE(oversized_finished);
    ASSERT_EQ(state.oversized_terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(state.oversized_terminals[0].event));
    const auto* oversized_error = std::get_if<Chorus::ChorusSignal::Error>(&state.oversized_terminals[0].event);
    ASSERT_TRUE(oversized_error != nullptr);
    if (oversized_error)
        ASSERT_EQ(oversized_error->code, Chorus::ChorusError::Decode);

    ASSERT_TRUE(recovery_finished);
    ASSERT_EQ(state.recovery_terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state.recovery_terminals[0].event));
    ASSERT_TRUE(state.recovery_tokens > 0);
}

TEST_F(LlamaSchedulerModelTest, Higher_priority_request_served_first) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests",
         int64_t{1}}, // one active request lets the priority queue choose before admission
    };

    std::mutex order_mutex;
    std::vector<int64_t> completion_order;

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config, {}).has_value());

    auto make_handler = [&](int64_t id) {
        return [&, id](const Chorus::ChorusSignal& sig) {
            if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) || std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
                std::lock_guard<std::mutex> lock(order_mutex);
                completion_order.push_back(id);
            }
        };
    };

    const std::string prompt = "<start_of_turn>user\nCount to five.<end_of_turn>\n<start_of_turn>model\n";

    Chorus::ChorusRequest low;
    low.id = 100;
    low.priority = 0;
    low.prompt = prompt;
    low.gen_config.max_tokens = 8;
    low.on_event = make_handler(100);

    Chorus::ChorusRequest high;
    high.id = 200;
    high.priority = 10;
    high.prompt = prompt;
    high.gen_config.max_tokens = 8;
    high.on_event = make_handler(200);

    // Best-effort: a narrow race exists if the worker ingests `low` in the sub-ms gap before `high` is queued.
    engine.submit_request(low);
    engine.submit_request(high);

    int timeout_ms = 30000;
    while (timeout_ms > 0) {
        {
            std::lock_guard<std::mutex> lock(order_mutex);
            if (completion_order.size() >= 2)
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        timeout_ms -= 50;
    }

    std::lock_guard<std::mutex> lock(order_mutex);
    ASSERT_EQ(completion_order.size(), 2);
    ASSERT_EQ(completion_order[0], 200); // high priority completes first

    engine.shutdown();
}

TEST_F(LlamaSchedulerModelTest, Sequence_id_reusable_after_request_completes) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{1}}, // force the second request to reuse the released sequence ID
    };

    struct Result {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> terminals;
        size_t tokens = 0;
    };

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config, {}).has_value());

    auto run_one = [&](int64_t id) {
        auto result = std::make_shared<Result>();

        Chorus::ChorusRequest req;
        req.id = id;
        req.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
        req.gen_config.max_tokens = 6;
        req.on_event = [result](const Chorus::ChorusSignal& sig) {
            std::lock_guard<std::mutex> lock(result->mutex);
            if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
                ++result->tokens;
            } else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
                       std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
                result->terminals.push_back(sig);
                result->cv.notify_all();
            }
        };

        engine.submit_request(req);

        bool finished = false;
        {
            std::unique_lock<std::mutex> lock(result->mutex);
            finished = result->cv.wait_for(lock, std::chrono::seconds(15), [&] {
                return !result->terminals.empty();
            });
        }
        return std::pair{result, finished};
    };

    auto [first, first_finished] = run_one(1);
    if (!first_finished) {
        engine.shutdown();
        ASSERT_TRUE(first_finished);
        return;
    }

    auto [second, second_finished] = run_one(2); // must reuse the reclaimed sequence ID

    engine.shutdown();

    ASSERT_TRUE(second_finished);
    for (const auto& result : {first, second}) {
        ASSERT_EQ(result->terminals.size(), size_t{1});
        ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(result->terminals[0].event));
        ASSERT_TRUE(result->tokens > 0);
    }
}

TEST_F(LlamaSchedulerModelTest, Batch_demand_beyond_capacity_is_clamped_not_overrun) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{64}},
        {"n_batch", int64_t{64}},
        {"n_ubatch", int64_t{32}},
        {"max_concurrent_requests", int64_t{4}},
    };

    // Long prompts keep every sequence in prefill across many passes, so several
    // requests contribute to the same batch regardless of ingest timing.
    std::string long_prompt;
    for (int i = 0; i < 40; ++i)
        long_prompt += "The quick brown fox jumps over the lazy dog. ";

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::map<int64_t, std::vector<Chorus::ChorusSignal>> terminals;
        size_t recovery_tokens = 0;
    } state;

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config, {}).has_value());

    for (int64_t id = 1; id <= 4; ++id) {
        Chorus::ChorusRequest req;
        req.id = id;
        req.prompt = long_prompt;
        req.gen_config.max_tokens = 4;
        req.on_event = [&](const Chorus::ChorusSignal& sig) {
            if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) &&
                !std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event))
                return;
            std::lock_guard<std::mutex> lock(state.mutex);
            state.terminals[sig.request_id].push_back(sig);
            state.cv.notify_all();
        };
        engine.submit_request(req);
    }

    bool oversized_finished = false;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        oversized_finished = state.cv.wait_for(lock, std::chrono::seconds(15), [&] {
            for (int64_t id = 1; id <= 4; ++id)
                if (state.terminals[id].empty())
                    return false;
            return true;
        });
    }
    if (!oversized_finished) {
        engine.shutdown();
        ASSERT_TRUE(oversized_finished);
        return;
    }

    // Engine must keep running: a fresh small request still completes.
    Chorus::ChorusRequest small;
    small.id = 5;
    small.prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    small.gen_config.max_tokens = 4;
    small.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    small.on_event = [&](const Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
            ++state.recovery_tokens;
        } else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
                   std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
            state.terminals[sig.request_id].push_back(sig);
            state.cv.notify_all();
        }
    };
    engine.submit_request(small);

    bool recovery_finished = false;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        recovery_finished = state.cv.wait_for(lock, std::chrono::seconds(15), [&] {
            return !state.terminals[small.id].empty();
        });
    }

    engine.shutdown();

    ASSERT_TRUE(oversized_finished);
    for (int64_t id = 1; id <= 4; ++id) {
        const auto& terminals = state.terminals[id];
        ASSERT_EQ(terminals.size(), size_t{1});
        ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(terminals[0].event));
        const auto* error = std::get_if<Chorus::ChorusSignal::Error>(&terminals[0].event);
        ASSERT_TRUE(error != nullptr);
        if (error)
            ASSERT_EQ(error->code, Chorus::ChorusError::Decode);
    }
    ASSERT_TRUE(recovery_finished);
    ASSERT_EQ(state.terminals[small.id].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state.terminals[small.id][0].event));
    ASSERT_TRUE(state.recovery_tokens > 0);
}
