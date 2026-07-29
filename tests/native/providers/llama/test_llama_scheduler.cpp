#include "chorus/core/common.hpp"
#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/providers/llama/llama_generation.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

static Chorus::ChorusConfig make_gguf_config(const std::string& path) {
    Chorus::ChorusConfig config;
    config.model.model_id = "test-model";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, path, std::nullopt, std::nullopt});
    return config;
}

static std::optional<Chorus::RequestRejection> load_rejection(const Chorus::ChorusConfig& config) {
    auto result = Chorus::parse_llama_load_config(config);
    if (const auto* rejection = std::get_if<Chorus::RequestRejection>(&result))
        return *rejection;
    return std::nullopt;
}

void test_load_option_defaults() {
    auto result = Chorus::parse_llama_load_config(make_gguf_config(MODEL_PATH));
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(result));
    const auto& load = std::get<Chorus::LlamaLoadConfig>(result);
    ASSERT_EQ(load.n_batch, uint32_t{2048});
    ASSERT_EQ(load.n_ubatch, uint32_t{512});
    ASSERT_EQ(load.main_gpu, int32_t{0});
    ASSERT_EQ(load.gpu_layers, int32_t{-1});
}

void test_load_option_accepts_exact_int64_values() {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"n_batch", int64_t{96}},
        {"n_ubatch", int64_t{32}},
        {"main_gpu", int64_t{2}},
    };
    auto result = Chorus::parse_llama_load_config(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(result));
    const auto& load = std::get<Chorus::LlamaLoadConfig>(result);
    ASSERT_EQ(load.n_batch, uint32_t{96});
    ASSERT_EQ(load.n_ubatch, uint32_t{32});
    ASSERT_EQ(load.main_gpu, int32_t{2});
}

void test_load_option_rejects_wrong_scalar_alternatives() {
    auto expect_rejection = [](const std::string& key, Chorus::ProviderOptionValue value) {
        auto config = make_gguf_config(MODEL_PATH);
        config.provider_options["llama"] = Chorus::ProviderOptionMap{{key, std::move(value)}};
        auto result = Chorus::parse_llama_load_config(config);
        const auto* rejection = std::get_if<Chorus::RequestRejection>(&result);
        return rejection && rejection->error == Chorus::ChorusError::UnsupportedOption &&
               rejection->message.find(key) != std::string::npos;
    };

    ASSERT_TRUE(expect_rejection("n_batch", true));
    ASSERT_TRUE(expect_rejection("n_ubatch", 32.0));
    ASSERT_TRUE(expect_rejection("main_gpu", "0"));
}

void test_load_option_rejects_invalid_batch_sizes() {
    for (const auto& [key, value] : std::vector<std::pair<std::string, int64_t>>{
             {"n_batch", 0}, {"n_batch", -1}, {"n_ubatch", 0}, {"n_ubatch", -1}
         }) {
        auto config = make_gguf_config(MODEL_PATH);
        config.provider_options["llama"] = Chorus::ProviderOptionMap{{key, value}};
        const auto rejection = load_rejection(config);
        ASSERT_TRUE(rejection.has_value());
        ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
        ASSERT_TRUE(rejection->message.find(key) != std::string::npos);
    }
}

void test_load_option_rejects_microbatch_larger_than_batch() {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"n_batch", int64_t{32}}, {"n_ubatch", int64_t{33}}};
    const auto rejection = load_rejection(config);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection->message.find("n_ubatch") != std::string::npos);
}

void test_load_option_rejects_negative_main_gpu() {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"main_gpu", int64_t{-1}}};
    const auto rejection = load_rejection(config);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection->message.find("main_gpu") != std::string::npos);
}

void test_load_option_rejects_narrowing_overflow() {
    auto expect_rejection = [](const std::string& key, int64_t value) {
        auto config = make_gguf_config(MODEL_PATH);
        config.provider_options["llama"] = Chorus::ProviderOptionMap{{key, value}};
        auto result = Chorus::parse_llama_load_config(config);
        const auto* rejection = std::get_if<Chorus::RequestRejection>(&result);
        return rejection && rejection->error == Chorus::ChorusError::UnsupportedOption &&
               rejection->message.find(key) != std::string::npos;
    };

    ASSERT_TRUE(expect_rejection("n_batch", int64_t{std::numeric_limits<int32_t>::max()} + 1));
    ASSERT_TRUE(expect_rejection("n_ubatch", int64_t{std::numeric_limits<uint32_t>::max()} + 1));
    ASSERT_TRUE(expect_rejection("main_gpu", int64_t{std::numeric_limits<int32_t>::max()} + 1));
}

void test_load_option_still_rejects_unknown_keys() {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"warp_factor", int64_t{9}}};
    const auto rejection = load_rejection(config);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    ASSERT_TRUE(rejection->message.find("warp_factor") != std::string::npos);
}

void test_load_option_rejects_cpu_with_explicit_gpu_controls() {
    auto expect_rejection = [](const std::string& key, int64_t value) {
        auto config = make_gguf_config(MODEL_PATH);
        config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}, {key, value}};
        const auto rejection = load_rejection(config);
        return rejection.has_value() && rejection->error == Chorus::ChorusError::UnsupportedOption &&
               rejection->message.find(key) != std::string::npos;
    };

    ASSERT_TRUE(expect_rejection("main_gpu", 0));
    ASSERT_TRUE(expect_rejection("gpu_layers", 17));
    ASSERT_TRUE(expect_rejection("gpu_layers", -1));
}

void test_load_option_accepts_cpu_with_explicit_zero_gpu_layers() {
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}, {"gpu_layers", int64_t{0}}};
    auto result = Chorus::parse_llama_load_config(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::LlamaLoadConfig>(result));
    const auto& load = std::get<Chorus::LlamaLoadConfig>(result);
    ASSERT_TRUE(load.gpu_layers_explicit);
    ASSERT_TRUE(!load.main_gpu_explicit);
    ASSERT_EQ(load.gpu_layers, int32_t{0});
}

void test_load_option_cpu_placement_disables_every_offload_path() {
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

void test_load_option_gpu_placement_preserves_upstream_device_selection() {
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

void test_load_option_context_params_forward_exact_values() {
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

void test_resolve_generation_unset_fields_use_upstream_defaults() {
    Chorus::GenerationConfig config; // everything unset
    auto result = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::ResolvedLlamaGeneration>(result));
    const auto& r = std::get<Chorus::ResolvedLlamaGeneration>(result);
    ASSERT_EQ(r.max_tokens, -1); // upstream n_predict default: unbounded
    ASSERT_EQ(r.sampling.top_k, 40);
    ASSERT_TRUE(r.sampling.top_p > 0.94f && r.sampling.top_p < 0.96f);
    ASSERT_TRUE(r.sampling.temp > 0.79f && r.sampling.temp < 0.81f);
    ASSERT_EQ(r.sampling.seed, LLAMA_DEFAULT_SEED);
    ASSERT_TRUE(r.sampling.penalty_repeat > 0.99f && r.sampling.penalty_repeat < 1.01f);
}

void test_resolve_generation_set_fields_override() {
    Chorus::GenerationConfig config;
    config.max_tokens = 32;
    config.temperature = 0.2f;
    config.top_k = 5;
    config.top_p = 0.5f;
    config.seed = uint64_t{7};
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"repeat_penalty", 1.3}};
    auto result = Chorus::resolve_llama_generation(config);
    ASSERT_TRUE(std::holds_alternative<Chorus::ResolvedLlamaGeneration>(result));
    const auto& r = std::get<Chorus::ResolvedLlamaGeneration>(result);
    ASSERT_EQ(r.max_tokens, 32);
    ASSERT_TRUE(r.sampling.temp > 0.19f && r.sampling.temp < 0.21f);
    ASSERT_EQ(r.sampling.top_k, 5);
    ASSERT_TRUE(r.sampling.top_p > 0.49f && r.sampling.top_p < 0.51f);
    ASSERT_EQ(r.sampling.seed, (uint32_t)7);
    ASSERT_TRUE(r.sampling.penalty_repeat > 1.29f && r.sampling.penalty_repeat < 1.31f);
}

void test_transient_decode_failure_recovers() {
    SKIP_IF_MODEL_TESTS_DISABLED();

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
    big.gen_config.max_tokens = 8;
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
    small.gen_config.max_tokens = 4;
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
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
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

    engine.stop();
}

void test_slot_reusable_after_request_completes() {
    SKIP_IF_MODEL_TESTS_DISABLED();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
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
        req.gen_config.max_tokens = 6;
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
        req.gen_config.max_tokens = 4;
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
    small.gen_config.max_tokens = 4;
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

    run_test("load option defaults", test_load_option_defaults);
    run_test("load option accepts exact int64 values", test_load_option_accepts_exact_int64_values);
    run_test("load option rejects wrong scalar alternatives", test_load_option_rejects_wrong_scalar_alternatives);
    run_test("load option rejects invalid batch sizes", test_load_option_rejects_invalid_batch_sizes);
    run_test("load option rejects n_ubatch above n_batch", test_load_option_rejects_microbatch_larger_than_batch);
    run_test("load option rejects negative main_gpu", test_load_option_rejects_negative_main_gpu);
    run_test("load option rejects narrowing overflow", test_load_option_rejects_narrowing_overflow);
    run_test("load option still rejects unknown keys", test_load_option_still_rejects_unknown_keys);
    run_test(
        "load option rejects CPU with explicit GPU controls", test_load_option_rejects_cpu_with_explicit_gpu_controls
    );
    run_test(
        "load option accepts CPU with explicit zero GPU layers",
        test_load_option_accepts_cpu_with_explicit_zero_gpu_layers
    );
    run_test(
        "load option CPU placement disables every offload path",
        test_load_option_cpu_placement_disables_every_offload_path
    );
    run_test(
        "load option GPU placement preserves upstream device selection",
        test_load_option_gpu_placement_preserves_upstream_device_selection
    );
    run_test("load option context params forward exact values", test_load_option_context_params_forward_exact_values);
    run_test(
        "resolve generation: unset fields use upstream defaults",
        test_resolve_generation_unset_fields_use_upstream_defaults
    );
    run_test("resolve generation: set fields override", test_resolve_generation_set_fields_override);
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
