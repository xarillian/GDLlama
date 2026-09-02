#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"
#include "gtest_utils.hpp"

class LlamaSchedulerModelTest : public ChorusModelTest {};

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

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

static Chorus::ChorusConfig make_gguf_config(const std::string& path) {
    Chorus::ChorusConfig config;
    config.model.model_id = "test-model";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, path});
    return config;
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

TEST(LlamaScheduler, load_option_context_params_forward_exact_values) {
    Chorus::LlamaLoadConfig load;
    load.context_size = 4096;
    load.num_slots = 3;
    load.thread_count = 6;
    load.n_batch = 96;
    load.n_ubatch = 32;
    const auto params = Chorus::make_llama_context_params(load);
    ASSERT_EQ(params.n_ctx, uint32_t{4096});
    ASSERT_EQ(params.n_seq_max, uint32_t{3});
    ASSERT_EQ(params.n_threads, int32_t{6});
    ASSERT_EQ(params.n_threads_batch, int32_t{6});
    ASSERT_EQ(params.n_batch, uint32_t{96});
    ASSERT_EQ(params.n_ubatch, uint32_t{32});
}

TEST_F(LlamaSchedulerModelTest, Transient_decode_failure_recovers) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{64}}, // tiny unified KV cache
        {"n_batch", int64_t{64}},      // keep decode-failure setup independent from the larger default batch
        {"n_ubatch", int64_t{32}},
        {"tokens_per_tick", int64_t{16}},
        {"num_slots", int64_t{1}},
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
        {"num_slots",
         int64_t{1}}, // one slot serializes execution; the priority_queue decides who runs first, not submit order
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

TEST_F(LlamaSchedulerModelTest, Slot_reusable_after_request_completes) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"num_slots", int64_t{1}}, // force the second request to reuse the first slot
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

    auto [second, second_finished] = run_one(2); // must reuse the reclaimed slot

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
        {"n_batch", int64_t{64}}, // explicit logical batch capacity
        {"n_ubatch", int64_t{32}},
        {"tokens_per_tick", int64_t{64}}, // per-slot demand; 4 slots * 64 = 256 tokens offered to a 64-token batch
        {"num_slots", int64_t{4}},
    };

    // Long prompts keep every slot in prefill for many ticks, so multiple slots
    // contribute to the same batch regardless of ingest timing.
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
