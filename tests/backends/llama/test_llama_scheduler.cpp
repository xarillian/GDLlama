#include "chorus/backends/llama/llama_engine.hpp"
#include "chorus/backends/llama/llama_utils.hpp"
#include "chorus/core/common.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

static Chorus::ChorusConfig make_gguf_config(const std::string& path) {
    Chorus::ChorusConfig config;
    config.model.model_id = "test-model";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({"weights", path, std::nullopt, std::nullopt});
    return config;
}

void test_resolve_sampling_unset_fields_use_upstream_defaults() {
    Chorus::GenerationConfig config; // everything unset
    auto r = Chorus::LlamaUtils::resolve_sampling(config);
    ASSERT_EQ(r.max_tokens, -1); // upstream n_predict default: unbounded
    ASSERT_EQ(r.top_k, 40);
    ASSERT_TRUE(r.top_p > 0.94f && r.top_p < 0.96f);
    ASSERT_TRUE(r.temperature > 0.79f && r.temperature < 0.81f);
    ASSERT_EQ(r.seed, LLAMA_DEFAULT_SEED);                             // unset seed means random
    ASSERT_TRUE(r.repeat_penalty > 0.99f && r.repeat_penalty < 1.01f); // upstream: disabled
}

void test_resolve_sampling_set_fields_override() {
    Chorus::GenerationConfig config;
    config.common.max_tokens = 32;
    config.common.temperature = 0.2f;
    config.common.top_k = 5;
    config.common.top_p = 0.5f;
    config.common.seed = uint64_t{7};
    config.backend_options["llama"] = Chorus::OptionMap{{"repeat_penalty", 1.3}};
    auto r = Chorus::LlamaUtils::resolve_sampling(config);
    ASSERT_EQ(r.max_tokens, 32);
    ASSERT_TRUE(r.temperature > 0.19f && r.temperature < 0.21f);
    ASSERT_EQ(r.top_k, 5);
    ASSERT_TRUE(r.top_p > 0.49f && r.top_p < 0.51f);
    ASSERT_EQ(r.seed, (uint32_t)7);
    ASSERT_TRUE(r.repeat_penalty > 1.29f && r.repeat_penalty < 1.31f);
}

void test_transient_decode_failure_recovers() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{64}},    // tiny unified KV cache
        {"tokens_per_tick", int64_t{16}}, // <= context_size so the batch never overruns before decode
        {"num_slots", int64_t{1}},
    };

    // A prompt guaranteed to exceed 64 KV cells (~2000 tokens).
    std::string huge_prompt;
    for (int i = 0; i < 200; ++i)
        huge_prompt += "The quick brown fox jumps over the lazy dog. ";

    std::atomic<bool> big_errored{false};
    Chorus::ChorusError big_code = Chorus::ChorusError::None;
    std::atomic<bool> small_done{false};
    std::atomic<int> small_tokens{0};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest big;
    big.id = 1;
    big.prompt = huge_prompt;
    big.gen_config.common.max_tokens = 8;
    big.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Error) {
            big_code = sig.error_code;
            big_errored = true;
        }
    };
    engine.submit_request(big);

    int timeout_ms = 15000;
    while (!big_errored && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        timeout_ms -= 50;
    }
    ASSERT_TRUE(big_errored);
    ASSERT_TRUE(big_code == Chorus::ChorusError::Decode);

    // Engine must keep running: a fresh small request still completes.
    Chorus::ChorusRequest small;
    small.id = 2;
    small.prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    small.gen_config.common.max_tokens = 4;
    small.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token)
            small_tokens++;
        else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error)
            small_done = true;
    };
    engine.submit_request(small);

    timeout_ms = 15000;
    while (!small_done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        timeout_ms -= 50;
    }
    ASSERT_TRUE(small_done);
    ASSERT_TRUE(small_tokens > 0);

    engine.stop();
}

void test_higher_priority_request_served_first() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"use_gpu", false},
        {"num_slots",
         int64_t{1}}, // one slot serializes execution; the priority_queue decides who runs first, not submit order
    };

    std::mutex order_mutex;
    std::vector<int64_t> completion_order;

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    auto make_handler = [&](int64_t id) {
        return [&, id](const Chorus::ChorusSignal& sig) {
            if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error) {
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
    low.gen_config.common.max_tokens = 8;
    low.on_event = make_handler(100);

    Chorus::ChorusRequest high;
    high.id = 200;
    high.priority = 10;
    high.prompt = prompt;
    high.gen_config.common.max_tokens = 8;
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

    engine.stop();
}

void test_slot_reusable_after_request_completes() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"use_gpu", false},
        {"num_slots", int64_t{1}}, // force the second request to reuse the first slot
    };

    std::atomic<int> tokens{0};
    std::atomic<bool> done{false};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    auto run_one = [&](int64_t id) -> int {
        tokens = 0;
        done = false;

        Chorus::ChorusRequest req;
        req.id = id;
        req.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
        req.gen_config.common.max_tokens = 6;
        req.on_event = [&](const Chorus::ChorusSignal& sig) {
            if (sig.type == Chorus::EventType::Token)
                tokens++;
            else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error)
                done = true;
        };

        engine.submit_request(req);

        int timeout_ms = 15000;
        while (!done && timeout_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            timeout_ms -= 50;
        }
        return done ? tokens.load() : -1;
    };

    int first = run_one(1);
    ASSERT_TRUE(first > 0); // completed and produced tokens

    int second = run_one(2); // must reuse the reclaimed slot
    ASSERT_TRUE(second > 0);

    engine.stop();
}

void test_batch_demand_beyond_capacity_is_clamped_not_overrun() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.backend_options["llama"] = Chorus::OptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{64}},    // batch capacity == context_size
        {"tokens_per_tick", int64_t{64}}, // per-slot demand; 4 slots * 64 = 256 tokens offered to a 64-token batch
        {"num_slots", int64_t{4}},
    };

    // Long prompts keep every slot in prefill for many ticks, so multiple slots
    // contribute to the same batch regardless of ingest timing.
    std::string long_prompt;
    for (int i = 0; i < 40; ++i)
        long_prompt += "The quick brown fox jumps over the lazy dog. ";

    std::atomic<int> terminal_signals{0};
    std::atomic<bool> small_done{false};
    std::atomic<int> small_tokens{0};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config).has_value());

    for (int64_t id = 1; id <= 4; ++id) {
        Chorus::ChorusRequest req;
        req.id = id;
        req.prompt = long_prompt;
        req.gen_config.common.max_tokens = 4;
        req.on_event = [&](const Chorus::ChorusSignal& sig) {
            if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error)
                terminal_signals++;
        };
        engine.submit_request(req);
    }

    int timeout_ms = 15000;
    while (terminal_signals < 4 && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        timeout_ms -= 50;
    }
    // Overcommitting the batch must resolve through the defined error path
    // (KV exhaustion -> Decode error), never a batch-buffer overrun.
    ASSERT_EQ(terminal_signals.load(), 4);

    // Engine must keep running: a fresh small request still completes.
    Chorus::ChorusRequest small;
    small.id = 5;
    small.prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    small.gen_config.common.max_tokens = 4;
    small.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (sig.type == Chorus::EventType::Token)
            small_tokens++;
        else if (sig.type == Chorus::EventType::Stop || sig.type == Chorus::EventType::Error)
            small_done = true;
    };
    engine.submit_request(small);

    timeout_ms = 15000;
    while (!small_done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        timeout_ms -= 50;
    }
    ASSERT_TRUE(small_done);
    ASSERT_TRUE(small_tokens > 0);

    engine.stop();
}

int run_llama_scheduler_tests() {
    std::cout << "\n--- LLAMA SCHEDULER SUITE ---\n";

    run_test(
        "resolve_sampling: unset fields use upstream defaults", test_resolve_sampling_unset_fields_use_upstream_defaults
    );
    run_test("resolve_sampling: set fields override", test_resolve_sampling_set_fields_override);
    run_test("Transient_decode_failure_recovers", test_transient_decode_failure_recovers);
    run_test("Higher_priority_request_served_first", test_higher_priority_request_served_first);
    run_test("Slot_reusable_after_request_completes", test_slot_reusable_after_request_completes);
    run_test(
        "Batch_demand_beyond_capacity_is_clamped_not_overrun", test_batch_demand_beyond_capacity_is_clamped_not_overrun
    );

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    } else {
        std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
        return 0;
    }
}
