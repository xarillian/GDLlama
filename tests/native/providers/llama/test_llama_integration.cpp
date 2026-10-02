#include "chorus/core/common.hpp"
#include "chorus/engine_factory.hpp"
#include "chorus/providers/llama/llama_engine.hpp"
#include "chorus/providers/llama/llama_generation_options.hpp"
#include "chorus/providers/llama/llama_load_config.hpp"
#include "chorus/runtime/runtime.hpp"
#include "collecting_log.hpp"
#include "gtest_utils.hpp"
#include "process_test.hpp"
#include "support/runtime_test_utils.hpp"

class LlamaIntegrationModelTest : public ChorusModelTest {};
class LlamaGpuModelTest : public ChorusGpuModelTest {};

#include <llama.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <string_view>
#include <thread>
#include <vector>

const std::string MODEL_PATH = "tests/models/gemma-3-270m-it-F16.gguf";

static Chorus::ChorusConfig make_gguf_config(const std::string& path) {
    Chorus::ChorusConfig config;
    config.model.model_id = "test-model";
    config.model.format = Chorus::ModelFormat::Gguf;
    config.model.assets.push_back({Chorus::AssetRole::Weights, path});
    return config;
}

// Everything the engine logs, vendor output included: one record per line,
// joined back into text for the substring oracles below. The engine owns
// llama's hook, so this reads the stream a real host reads, not a private tap.
class EngineLogCapture {
  public:
    Chorus::Logger logger() { return _logs.logger(); }

    std::string text() const {
        std::string text;
        for (const auto& record : _logs.records()) {
            text += record.message;
            text += '\n';
        }
        return text;
    }

  private:
    CollectingLog _logs;
};

bool contains_vulkan_compute_buffer_log(std::string_view logs) {
    size_t line_start = 0;
    while (line_start < logs.size()) {
        const size_t line_end = logs.find('\n', line_start);
        const std::string_view line = logs.substr(
            line_start, line_end == std::string_view::npos ? logs.size() - line_start : line_end - line_start
        );
        if (line.find("Vulkan") != std::string_view::npos &&
            line.find("compute buffer size") != std::string_view::npos) {
            return true;
        }
        if (line_end == std::string_view::npos)
            break;
        line_start = line_end + 1;
    }
    return false;
}

TEST(LlamaIntegration, Llama_unsupported_model_format_is_rejected) {
    // No model needed: the format gate fires before any file I/O.
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config("/nonexistent.safetensors");
    config.model.format = Chorus::ModelFormat::SafeTensors;
    auto err = engine.initialize(config, {}, {});
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(*err == Chorus::ChorusError::UnsupportedModelFormat);
}

TEST(LlamaIntegration, Llama_unknown_load_option_is_rejected) {
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH); // existing macro/constant in this file
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"warp_factor", int64_t{9}}};
    auto err = engine.initialize(config, {}, {});
    ASSERT_TRUE(err.has_value());
    ASSERT_TRUE(*err == Chorus::ChorusError::UnsupportedOption);
}

TEST_F(LlamaIntegrationModelTest, Llama_load_lifecycle_reports_model_facts_and_prompt_rendering) {
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"context_size", int64_t{1024}},
        {"use_gpu", false},
    };

    ASSERT_TRUE(!engine.loaded_model_info().has_value());
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());
    ASSERT_TRUE(engine.is_initialized());
    ASSERT_TRUE(engine.capabilities().prompt_rendering);

    const auto info = engine.loaded_model_info();
    ASSERT_TRUE(info.has_value());
    if (info) {
        ASSERT_EQ(info->model_id, std::string("test-model"));
        ASSERT_EQ(info->format, Chorus::ModelFormat::Gguf);
        ASSERT_TRUE(!info->family.empty());
        ASSERT_TRUE(info->maximum_context.has_value() && *info->maximum_context > 0);
        ASSERT_TRUE(info->per_request_context.has_value() && *info->per_request_context > 0);
        ASSERT_TRUE(info->model_bytes.has_value() && *info->model_bytes > 0);
    }

    engine.shutdown();
    ASSERT_TRUE(!engine.is_initialized());
    ASSERT_TRUE(!engine.loaded_model_info().has_value());
}

TEST_F(LlamaIntegrationModelTest, Llama_repeated_healthy_initialization_changes_no_engine_state) {

    EngineLogCapture original_logs;
    EngineLogCapture replacement_logs;
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    ASSERT_TRUE(!engine.initialize(config, original_logs.logger(), {}).has_value());
    const auto before = engine.loaded_model_info();
    ASSERT_TRUE(before.has_value());

    Chorus::ChorusConfig replacement = make_gguf_config("/unused/replacement.gguf");
    replacement.model.model_id = "unused-replacement";
    replacement.model.format = Chorus::ModelFormat::SafeTensors;
    ASSERT_TRUE(!engine.initialize(replacement, replacement_logs.logger(), {}).has_value());
    const auto after = engine.loaded_model_info();
    ASSERT_TRUE(after.has_value());
    ASSERT_EQ(after->model_id, before->model_id);
    ASSERT_TRUE(after->format == before->format);
    ASSERT_EQ(after->family, before->family);
    ASSERT_EQ(after->quantization, before->quantization);

    engine.shutdown();
    Chorus::ChorusRequest stopped_request;
    stopped_request.id = 9010;
    stopped_request.prompt = "not accepted";
    engine.submit_request(stopped_request);

    ASSERT_TRUE(replacement_logs.text().empty());
    ASSERT_TRUE(original_logs.text().find("Request submitted to a stopped engine") != std::string::npos);
}

TEST_F(LlamaIntegrationModelTest, Llama_CPU_placement_avoids_Vulkan_compute_buffer) {

    EngineLogCapture log_capture;
    Chorus::LlamaEngine engine;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    ASSERT_TRUE(!engine.initialize(config, log_capture.logger(), {}).has_value());

    std::mutex mutex;
    std::condition_variable cv;
    int token_count = 0;
    int terminal_count = 0;
    bool errored = false;

    Chorus::ChorusRequest request;
    request.id = 9001;
    request.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    request.gen_config.max_tokens = 1;
    request.on_event = [&](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            ++token_count;
        } else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
        ) {
            ++terminal_count;
            errored = std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
            cv.notify_one();
        }
    };
    engine.submit_request(request);

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return terminal_count == 1; }));
    }
    engine.shutdown();

    ASSERT_EQ(token_count, 1);
    ASSERT_EQ(terminal_count, 1);
    ASSERT_TRUE(!errored);

    const std::string logs = log_capture.text();
    ASSERT_TRUE(!contains_vulkan_compute_buffer_log(logs));
    ASSERT_TRUE(logs.find("CPU compute buffer size") != std::string::npos);
}

TEST_F(LlamaGpuModelTest, Llama_GPU_placement_uses_Vulkan_compute_buffer) {
    EngineLogCapture log_capture;
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", true}};

    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, log_capture.logger(), {}).has_value());
    engine.shutdown();

    ASSERT_TRUE(contains_vulkan_compute_buffer_log(log_capture.text())) << log_capture.text();
}

TEST_F(LlamaIntegrationModelTest, Llama_batch_controls_create_context_and_generate_four_tokens) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"n_batch", int64_t{96}},
        {"n_ubatch", int64_t{32}},
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::ChorusSignal> terminals;
    size_t token_chunks = 0;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest request;
    request.id = 96;
    request.prompt = "<start_of_turn>user\nSay hello.<end_of_turn>\n<start_of_turn>model\n";
    request.gen_config.max_tokens = 4;
    request.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    request.on_event = [&](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            ++token_chunks;
        } else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
        ) {
            terminals.push_back(signal);
            cv.notify_all();
        }
    };
    engine.submit_request(request);

    bool completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        completed = cv.wait_for(lock, std::chrono::seconds(15), [&] { return !terminals.empty(); });
    }
    engine.shutdown();

    ASSERT_TRUE(completed);
    ASSERT_EQ(terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(terminals[0].event));
    ASSERT_EQ(token_chunks, size_t{4});
}

TEST_F(LlamaIntegrationModelTest, Llama_effective_batch_capacity_contains_oversized_prompt_failure) {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{128}},
        {"n_batch", int64_t{512}},
        {"n_ubatch", int64_t{128}},
    };

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> oversized_terminals;
        std::vector<Chorus::ChorusSignal> recovery_terminals;
    } state;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());
    const auto model_info = engine.loaded_model_info();
    ASSERT_TRUE(model_info && model_info->per_request_context);

    Chorus::ChorusRequest request;
    request.id = 512;
    for (int i = 0; i < 600; ++i)
        request.prompt += "hello ";
    request.gen_config.max_tokens = 1;
    request.on_event = [&](const Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.oversized_terminals.push_back(signal);
        state.cv.notify_all();
    };
    engine.submit_request(request);

    bool oversized_finished = false;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        oversized_finished =
            state.cv.wait_for(lock, std::chrono::seconds(15), [&] { return !state.oversized_terminals.empty(); });
    }
    if (!oversized_finished) {
        engine.shutdown();
        ASSERT_TRUE(oversized_finished);
        return;
    }

    Chorus::ChorusRequest recovery;
    recovery.id = 513;
    recovery.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    recovery.gen_config.max_tokens = 1;
    recovery.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    recovery.on_event = [&](const Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.recovery_terminals.push_back(signal);
        state.cv.notify_all();
    };
    engine.submit_request(recovery);

    bool recovery_finished = false;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        recovery_finished =
            state.cv.wait_for(lock, std::chrono::seconds(15), [&] { return !state.recovery_terminals.empty(); });
    }
    engine.shutdown();

    ASSERT_TRUE(oversized_finished);
    ASSERT_EQ(state.oversized_terminals.size(), size_t{1});
    const auto* oversized_error = std::get_if<Chorus::ChorusSignal::Error>(&state.oversized_terminals[0].event);
    ASSERT_TRUE(oversized_error != nullptr);
    if (oversized_error) {
        ASSERT_EQ(oversized_error->code, Chorus::ChorusError::Decode);
        ASSERT_NE(oversized_error->message.find("max_concurrent_requests=1"), std::string::npos);
        ASSERT_NE(
            oversized_error->message.find(
                "effective per-request context " + std::to_string(*model_info->per_request_context)
            ),
            std::string::npos
        );
    }
    ASSERT_TRUE(recovery_finished);
    ASSERT_EQ(state.recovery_terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state.recovery_terminals[0].event));
}

TEST_F(LlamaIntegrationModelTest, Llama_concurrent_requests_complete_with_multiple_sequences) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{2}},
    };

    std::atomic<int> completed_count{0};
    std::string responses[2];
    std::mutex responses_mutex;

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    for (int request_index = 0; request_index < 2; ++request_index) {
        Chorus::ChorusRequest request;
        request.id = request_index;
        request.prompt = "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n";
        request.gen_config.max_tokens = 10;
        request.on_event = [&, request_index](const Chorus::ChorusSignal& sig) {
            if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
                std::lock_guard<std::mutex> lock(responses_mutex);
                responses[request_index] += std::get<Chorus::ChorusSignal::Token>(sig.event).text;
            } else if (
                std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
                std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)
            ) {
                completed_count++;
            }
        };
        engine.submit_request(request);
    }

    int timeout_ms = 30000;
    while (completed_count < 2 && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    if (timeout_ms <= 0) {
        ADD_FAILURE() << "Timed out waiting for concurrent generation.";
        return;
    }

    ASSERT_TRUE(!responses[0].empty());
    ASSERT_TRUE(!responses[1].empty());
}

TEST_F(LlamaIntegrationModelTest, Max_tokens_counts_generated_not_prompt_tokens) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{1024}},
    };

    std::atomic<int> token_count{0};
    std::atomic<bool> done{false};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    std::string long_prompt = "<start_of_turn>user\n";
    for (int i = 0; i < 40; ++i)
        long_prompt += "Tell me a long and detailed story about dragons and castles. ";
    long_prompt += "<end_of_turn>\n<start_of_turn>model\n";

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = long_prompt;
    req.gen_config.max_tokens = 8;
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event))
            token_count++;
        else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)
        )
            done = true;
    };

    engine.submit_request(req);

    int timeout_ms = 15000;
    while (!done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    ASSERT_TRUE(done);
    ASSERT_TRUE(token_count > 1);
    ASSERT_TRUE(token_count <= 8);
}

TEST_F(LlamaIntegrationModelTest, Engine_reinitializes_and_generates_after_shutdown) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};

    std::atomic<int> tokens{0};
    std::atomic<bool> done{false};

    // declared after the state its worker callbacks capture, so the engine (and its worker thread) is destroyed first
    Chorus::LlamaEngine engine;

    std::stop_source stop;
    bool saw_weight_progress = false;
    Chorus::InitializationControl control{stop.get_token(), [&](const Chorus::LoadProgress& progress) {
                                              if (progress.phase == Chorus::LoadPhase::LoadingModel &&
                                                  progress.fraction) {
                                                  saw_weight_progress = true;
                                                  stop.request_stop();
                                              }
                                          }};
    const auto cancelled = engine.initialize(config, {}, control);
    ASSERT_TRUE(saw_weight_progress);
    ASSERT_TRUE(cancelled.has_value());
    ASSERT_EQ(cancelled->error, Chorus::ChorusError::Cancelled);
    ASSERT_FALSE(engine.is_initialized());
    engine.shutdown();
    ASSERT_TRUE(!engine.is_initialized());

    // Re-init must rebuild cleanly (robust shutdown() freed everything) and still generate.
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());
    ASSERT_TRUE(engine.is_initialized());

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    req.gen_config.max_tokens = 5;
    req.on_event = [&](const Chorus::ChorusSignal& sig) {
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event))
            tokens++;
        else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)
        )
            done = true;
    };
    engine.submit_request(req);

    int timeout_ms = 15000;
    while (!done && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timeout_ms -= 100;
    }

    ASSERT_TRUE(done);
    ASSERT_TRUE(tokens > 0);
    engine.shutdown();
}

TEST(LlamaIntegration, Llama_constraint_capabilities) {
    Chorus::LlamaEngine engine; // pre-init envelope
    auto caps = engine.capabilities();
    ASSERT_EQ(caps.provider_id, std::string("llama"));
    ASSERT_TRUE(caps.scheduling == Chorus::SchedulingAuthority::ChorusManaged);
    ASSERT_EQ(caps.model_formats.size(), (size_t)1);
    ASSERT_TRUE(caps.model_formats[0] == Chorus::ModelFormat::Gguf);
    ASSERT_EQ(caps.constraint_formats.size(), size_t{2});
    ASSERT_TRUE(caps.constraint_formats[0] == Chorus::ConstraintFormat::Gbnf);
    ASSERT_TRUE(caps.constraint_formats[1] == Chorus::ConstraintFormat::JsonSchema);
    ASSERT_EQ(caps.common_generation_options.size(), (size_t)10);
    ASSERT_TRUE(
        std::find(caps.common_generation_options.begin(), caps.common_generation_options.end(), "constraint") !=
        caps.common_generation_options.end()
    );
    ASSERT_TRUE(
        std::find(caps.common_generation_options.begin(), caps.common_generation_options.end(), "stop") !=
        caps.common_generation_options.end()
    );
    ASSERT_TRUE(caps.cancellation);
}

TEST_F(LlamaIntegrationModelTest, Llama_cancellation_removes_queued_request_before_active_finishes) {

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool active_started = false;
        bool active_terminal = false;
        bool active_terminal_when_queued_cancelled = false;
        std::vector<Chorus::ChorusSignal> queued_signals;
    };
    auto state = std::make_shared<State>();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest active;
    active.id = 301;
    active.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
    active.gen_config.max_tokens = 512;
    active.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    active.on_event = [state](Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            state->active_started = true;
            state->cv.notify_all();
        } else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
        ) {
            state->active_terminal = true;
            state->cv.notify_all();
        }
    };
    engine.submit_request(active);

    bool active_started = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        active_started = state->cv.wait_for(lock, std::chrono::seconds(15), [&] { return state->active_started; });
    }
    if (!active_started) {
        engine.shutdown();
        ASSERT_TRUE(active_started);
    }

    Chorus::ChorusRequest queued;
    queued.id = 302;
    queued.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    queued.gen_config.max_tokens = 2;
    queued.on_event = [state](Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->queued_signals.push_back(signal);
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)) {
            state->active_terminal_when_queued_cancelled = state->active_terminal;
            state->cv.notify_all();
        }
    };
    engine.submit_request(queued);
    engine.cancel_request(queued.id);

    bool queued_terminal = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        queued_terminal = state->cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return !state->queued_signals.empty() &&
                   (std::holds_alternative<Chorus::ChorusSignal::Stop>(state->queued_signals.back().event) ||
                    std::holds_alternative<Chorus::ChorusSignal::Error>(state->queued_signals.back().event));
        });
    }
    engine.shutdown();

    ASSERT_TRUE(queued_terminal);
    ASSERT_EQ(state->queued_signals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(state->queued_signals[0].event));
    ASSERT_TRUE(
        std::get<Chorus::ChorusSignal::Error>(state->queued_signals[0].event).code == Chorus::ChorusError::Cancelled
    );
    ASSERT_EQ(state->queued_signals[0].request_id, queued.id);
    ASSERT_TRUE(!state->active_terminal_when_queued_cancelled);
}

TEST_F(LlamaIntegrationModelTest, Llama_cancellation_is_idempotent_and_releases_active_slot) {

    std::mutex mutex;
    std::condition_variable cv;
    bool active_started = false;
    bool reuse_stopped = false;
    std::vector<Chorus::ChorusSignal> active_signals;

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest active;
    active.id = 311;
    active.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
    active.gen_config.max_tokens = 512;
    active.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        active_signals.push_back(signal);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event))
            active_started = true;
        cv.notify_all();
    };
    engine.submit_request(active);

    bool started = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        started = cv.wait_for(lock, std::chrono::seconds(15), [&] { return active_started; });
    }
    if (!started) {
        engine.shutdown();
        ASSERT_TRUE(started);
    }

    for (int i = 0; i < 8; ++i)
        engine.cancel_request(active.id);
    for (int64_t unknown = 9000; unknown < 9020; ++unknown)
        engine.cancel_request(unknown);

    bool active_terminal = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        active_terminal = cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return std::count_if(active_signals.begin(), active_signals.end(), [](const auto& signal) {
                       return std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
                              std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
                   }) == 1;
        });
    }
    if (!active_terminal) {
        engine.shutdown();
        ASSERT_TRUE(active_terminal);
    }

    Chorus::ChorusRequest reuse;
    reuse.id = 312;
    reuse.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    reuse.gen_config.max_tokens = 2;
    reuse.on_event = [&](Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::lock_guard<std::mutex> lock(mutex);
        reuse_stopped = std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event);
        cv.notify_all();
    };
    engine.submit_request(reuse);

    bool reuse_completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        reuse_completed = cv.wait_for(lock, std::chrono::seconds(15), [&] { return reuse_stopped; });
    }
    engine.shutdown();

    size_t terminal_count = 0;
    Chorus::ChorusError terminal_code = Chorus::ChorusError::None;
    for (const auto& signal : active_signals) {
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)) {
            ++terminal_count;
            terminal_code = std::get<Chorus::ChorusSignal::Error>(signal.event).code;
        }
    }
    ASSERT_TRUE(active_terminal);
    ASSERT_EQ(terminal_count, size_t{1});
    ASSERT_TRUE(terminal_code == Chorus::ChorusError::Cancelled);
    ASSERT_TRUE(reuse_completed);
}

TEST_F(LlamaIntegrationModelTest, Llama_cancellation_from_committed_buffered_token_does_not_replace_Stop) {

    struct Result {
        std::mutex mutex;
        std::condition_variable cv;
        std::string text;
        std::vector<Chorus::ChorusSignal> terminals;
    };

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto run = [&](int64_t id, std::vector<std::string> stop, bool cancel_from_token) {
        auto result = std::make_shared<Result>();
        Chorus::ChorusRequest request;
        request.id = id;
        request.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
        request.gen_config.max_tokens = 1;
        request.gen_config.seed = 42;
        request.gen_config.temperature = 0.0f;
        request.gen_config.stop = std::move(stop);
        request.on_event = [&, result, id, cancel_from_token](Chorus::ChorusSignal& signal) {
            {
                std::lock_guard<std::mutex> lock(result->mutex);
                if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event))
                    result->text += std::get<Chorus::ChorusSignal::Token>(signal.event).text;
                else if (
                    std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
                    std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
                )
                    result->terminals.push_back(signal);
            }
            if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event) && cancel_from_token)
                engine.cancel_request(id);
            result->cv.notify_all();
        };
        engine.submit_request(request);

        std::unique_lock<std::mutex> lock(result->mutex);
        const bool terminal =
            result->cv.wait_for(lock, std::chrono::seconds(15), [&] { return !result->terminals.empty(); });
        return std::pair{result, terminal};
    };

    auto [baseline, baseline_terminal] = run(326, {}, false);
    bool baseline_has_text = false;
    std::string buffered_marker;
    {
        std::lock_guard<std::mutex> lock(baseline->mutex);
        baseline_has_text = !baseline->text.empty();
        buffered_marker = baseline->text + "<UNMATCHED>";
    }
    if (!baseline_terminal || !baseline_has_text) {
        engine.shutdown();
        ASSERT_TRUE(baseline_terminal);
        ASSERT_TRUE(baseline_has_text);
    }

    auto [reentrant, reentrant_terminal] = run(327, {buffered_marker}, true);
    engine.shutdown();

    ASSERT_TRUE(reentrant_terminal);
    ASSERT_EQ(reentrant->terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(reentrant->terminals[0].event));
}

TEST_F(LlamaIntegrationModelTest, Llama_shutdown_waits_for_active_cancellation_callback_and_drains_queue) {

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool active_started = false;
        bool release_cancellation = false;
        bool cancellation_entered = false;
        bool stop_started = false;
        bool stop_returned = false;
        std::vector<Chorus::ChorusSignal> terminals;
    };
    auto state = std::make_shared<State>();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto callback = [state](Chorus::ChorusSignal& signal) {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            state->active_started = true;
            state->cv.notify_all();
            return;
        }
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        state->terminals.push_back(signal);
        if (signal.request_id == 331 && std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event) &&
            std::get<Chorus::ChorusSignal::Error>(signal.event).code == Chorus::ChorusError::Cancelled &&
            !state->cancellation_entered) {
            state->cancellation_entered = true;
            state->cv.notify_all();
            state->cv.wait(lock, [&] { return state->release_cancellation; });
        }
    };

    Chorus::ChorusRequest active;
    active.id = 331;
    active.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
    active.gen_config.max_tokens = 512;
    active.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    active.on_event = callback;
    engine.submit_request(active);

    bool started = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        started = state->cv.wait_for(lock, std::chrono::seconds(15), [&] { return state->active_started; });
    }
    if (!started) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->release_cancellation = true;
            state->cv.notify_all();
        }
        engine.shutdown();
        ASSERT_TRUE(started);
    }

    Chorus::ChorusRequest queued;
    queued.id = 332;
    queued.prompt = "queued request";
    queued.gen_config.max_tokens = 2;
    queued.on_event = callback;
    engine.submit_request(queued);

    engine.cancel_request(active.id);
    bool cancellation_callback_entered = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        cancellation_callback_entered =
            state->cv.wait_for(lock, std::chrono::seconds(5), [&] { return state->cancellation_entered; });
        if (!cancellation_callback_entered) {
            state->release_cancellation = true;
            state->cv.notify_all();
        }
    }
    if (!cancellation_callback_entered) {
        engine.shutdown();
        ASSERT_TRUE(cancellation_callback_entered);
    }

    std::thread stopper([&] {
        {
            std::lock_guard<std::mutex> state_lock(state->mutex);
            state->stop_started = true;
            state->cv.notify_all();
        }
        engine.shutdown();
        std::lock_guard<std::mutex> state_lock(state->mutex);
        state->stop_returned = true;
        state->cv.notify_all();
    });

    bool stop_started = false;
    bool returned_while_blocked = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        stop_started = state->cv.wait_for(lock, std::chrono::seconds(5), [&] { return state->stop_started; });
        if (stop_started) {
            returned_while_blocked =
                state->cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return state->stop_returned; });
        }
        state->release_cancellation = true;
        state->cv.notify_all();
    }
    stopper.join();

    size_t active_cancelled = 0;
    size_t queued_cancelled = 0;
    for (const auto& signal : state->terminals) {
        ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event));
        ASSERT_TRUE(std::get<Chorus::ChorusSignal::Error>(signal.event).code == Chorus::ChorusError::Cancelled);
        active_cancelled += signal.request_id == active.id;
        queued_cancelled += signal.request_id == queued.id;
    }
    ASSERT_TRUE(cancellation_callback_entered);
    ASSERT_TRUE(stop_started);
    ASSERT_TRUE(!returned_while_blocked);
    ASSERT_TRUE(state->stop_returned);
    ASSERT_EQ(active_cancelled, size_t{1});
    ASSERT_EQ(queued_cancelled, size_t{1});
}

TEST_F(LlamaIntegrationModelTest, Llama_rejects_unwired_controls_explicitly) {
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_gguf_config(MODEL_PATH), {}, {}).has_value());
    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "hi";
    req.gen_config.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= \"x\""};
    auto r = engine.validate_request(req);
    ASSERT_TRUE(!r.has_value());

    req.gen_config.constraint.reset();
    req.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"mirostat", int64_t{2}}};
    auto r2 = engine.validate_request(req);
    ASSERT_TRUE(!r2.has_value());

    // A known key with the wrong value type must reject.
    req.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"repeat_penalty", int64_t{2}}};
    auto r3 = engine.validate_request(req);
    ASSERT_TRUE(r3.has_value());
    ASSERT_TRUE(r3->error == Chorus::ChorusError::UnsupportedOption);
    engine.shutdown();
}

namespace {

struct ConstraintRequestResult {
    bool completed = false;
    bool timed_out = false;
    std::string response;
    std::optional<Chorus::ChorusError> error;
    std::string terminal_error;
};

struct ConstraintRequestState {
    std::mutex mutex;
    std::condition_variable cv;
    ConstraintRequestResult result;
};

void print_constraint_failure(const ConstraintRequestResult& result) {
    std::cerr << "[CONSTRAINT RESPONSE] " << nlohmann::ordered_json(result.response).dump() << "\n";
    std::cerr << "[CONSTRAINT TERMINAL ERROR] "
              << (result.terminal_error.empty() ? "No terminal error was emitted." : result.terminal_error) << "\n";
}

ConstraintRequestResult
run_constraint_request(Chorus::LlamaEngine& engine, int64_t request_id, Chorus::OutputConstraint constraint) {
    auto state = std::make_shared<ConstraintRequestState>();

    Chorus::ChorusRequest request;
    request.id = request_id;
    request.prompt = "<start_of_turn>user\nRespond now.<end_of_turn>\n<start_of_turn>model\n";
    request.gen_config.seed = 42;
    request.gen_config.temperature = 0.0f;
    request.gen_config.max_tokens = 32;
    request.gen_config.constraint = std::move(constraint);
    request.on_event = [state](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            state->result.response += std::get<Chorus::ChorusSignal::Token>(signal.event).text;
        } else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event)) {
            state->result.completed = true;
            state->cv.notify_one();
        } else if (std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)) {
            state->result.error = std::get<Chorus::ChorusSignal::Error>(signal.event).code;
            state->result.terminal_error = std::get<Chorus::ChorusSignal::Error>(signal.event).message;
            state->cv.notify_one();
        }
    };
    engine.submit_request(request);

    std::unique_lock<std::mutex> lock(state->mutex);
    if (!state->cv.wait_for(lock, std::chrono::seconds(30), [&] {
            return state->result.completed || state->result.error.has_value();
        })) {
        state->result.timed_out = true;
        state->result.terminal_error = "Timed out waiting for a terminal event.";
    }
    if (state->result.timed_out || state->result.error)
        std::cerr << "[CONSTRAINT TERMINAL ERROR] " << state->result.terminal_error << "\n";
    return state->result;
}

} // namespace

TEST_F(LlamaIntegrationModelTest, Llama_GBNF_constraint_enforces_output) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto result = run_constraint_request(
        engine, 101, Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, R"(root ::= "\"PINK_MOTH\"")"}
    );
    engine.shutdown();

    ASSERT_TRUE(!result.timed_out);
    ASSERT_TRUE(!result.error.has_value());
    ASSERT_TRUE(result.completed);
    if (result.response != "\"PINK_MOTH\"")
        print_constraint_failure(result);
    ASSERT_EQ(result.response, std::string("\"PINK_MOTH\""));
}

TEST_F(LlamaIntegrationModelTest, Llama_JSON_Schema_constraint_enforces_output) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto result = run_constraint_request(
        engine,
        102,
        Chorus::OutputConstraint{
            Chorus::ConstraintFormat::JsonSchema,
            R"({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false})",
        }
    );
    engine.shutdown();

    ASSERT_TRUE(!result.timed_out);
    ASSERT_TRUE(!result.error.has_value());
    ASSERT_TRUE(result.completed);
    const auto parsed = nlohmann::ordered_json::parse(result.response, nullptr, false);
    if (parsed.is_discarded())
        print_constraint_failure(result);
    ASSERT_TRUE(!parsed.is_discarded());
    ASSERT_TRUE(parsed.contains("ok"));
    ASSERT_TRUE(parsed["ok"].is_boolean());
}

TEST_F(LlamaIntegrationModelTest, Llama_invalid_grammar_is_isolated_to_one_constraint_request) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto invalid =
        run_constraint_request(engine, 103, Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= ["});
    ASSERT_TRUE(!invalid.timed_out);
    ASSERT_TRUE(invalid.error == Chorus::ChorusError::InvalidRequest);
    ASSERT_EQ(invalid.terminal_error, std::string("Invalid GBNF constraint: failed to parse grammar"));

    auto valid = run_constraint_request(
        engine, 104, Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, R"(root ::= "\"PINK_MOTH\"")"}
    );
    engine.shutdown();

    ASSERT_TRUE(!valid.timed_out);
    ASSERT_TRUE(!valid.error.has_value());
    ASSERT_TRUE(valid.completed);
    if (valid.response != "\"PINK_MOTH\"")
        print_constraint_failure(valid);
    ASSERT_EQ(valid.response, std::string("\"PINK_MOTH\""));
}

// Conformance: every option listed as supported provably does something.
// seed+temperature: same set seed twice => identical text. top_k: =1 forces
// greedy => deterministic without a seed. max_tokens: bounds output length.
// top_p: forwarded to the chain, but its behavioral distinctness alone is not
// stable enough to assert on a 270M model; top_k and top_p set-field
// forwarding is proven by the resolved generation override test in
// test_llama_scheduler.cpp instead.
TEST_F(LlamaIntegrationModelTest, Llama_conformance_seed_and_temperature) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};

    auto run_once = [&](std::string& text) {
        std::mutex sig_mutex;
        std::condition_variable cv;
        bool done = false;

        // declared after the state its worker callbacks capture, so the engine
        // (and its worker thread) is destroyed first
        Chorus::LlamaEngine engine;
        ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

        Chorus::ChorusRequest req;
        req.id = 1;
        req.prompt = "<start_of_turn>user\nTell me about dragons.<end_of_turn>\n<start_of_turn>model\n";
        req.gen_config.seed = 42;
        req.gen_config.temperature = 0.9f;
        req.gen_config.max_tokens = 24;
        req.on_event = [&](const Chorus::ChorusSignal& sig) {
            std::lock_guard<std::mutex> lock(sig_mutex);
            if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event)) {
                text += std::get<Chorus::ChorusSignal::Token>(sig.event).text;
            } else if (
                std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event) ||
                std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)
            ) {
                done = true;
                cv.notify_one();
            }
        };
        engine.submit_request(req);

        {
            std::unique_lock<std::mutex> lock(sig_mutex);
            cv.wait_for(lock, std::chrono::seconds(15), [&] { return done; });
        }
        engine.shutdown();
    };

    std::string text_a;
    std::string text_b;
    run_once(text_a);
    run_once(text_b);
    ASSERT_TRUE(!text_a.empty());
    ASSERT_EQ(text_a, text_b);
}

TEST_F(LlamaIntegrationModelTest, Llama_stop_zero_tokens_completes_and_reuses_slot) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{1}},
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::ChorusSignal::Event> zero_events;
    std::vector<Chorus::ChorusSignal::Event> reuse_events;
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest zero;
    zero.id = 201;
    zero.prompt = "This prompt must not enter inference.";
    zero.gen_config.max_tokens = 0;
    zero.on_event = [&](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        zero_events.push_back(signal.event);
        cv.notify_one();
    };
    engine.submit_request(zero);

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(5), [&] { return !zero_events.empty(); }));
        ASSERT_EQ(zero_events.size(), size_t{1});
        ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(zero_events[0]));
    }

    Chorus::ChorusRequest reuse;
    reuse.id = 202;
    reuse.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    reuse.gen_config.max_tokens = 2;
    reuse.on_event = [&](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        reuse_events.push_back(signal.event);
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            cv.notify_one();
    };
    engine.submit_request(reuse);

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(15), [&] {
            return !reuse_events.empty() && (std::holds_alternative<Chorus::ChorusSignal::Stop>(reuse_events.back()) ||
                                             std::holds_alternative<Chorus::ChorusSignal::Error>(reuse_events.back()));
        }));
        ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(reuse_events.back()));
        ASSERT_TRUE(std::any_of(reuse_events.begin(), reuse_events.end(), [](const auto& event) {
            return std::holds_alternative<Chorus::ChorusSignal::Token>(event);
        }));
    }
    engine.shutdown();
}

TEST_F(LlamaIntegrationModelTest, Llama_stop_marker_never_emits_and_slot_reuses) {

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"max_concurrent_requests", int64_t{1}},
    };

    struct Result {
        std::vector<std::string> chunks;
        bool stopped = false;
        bool errored = false;
        bool timed_out = false;
    };

    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto run = [&](int64_t id, const std::vector<std::string>& stops, int32_t max_tokens) {
        auto result = std::make_shared<Result>();
        auto mutex = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();

        Chorus::ChorusRequest request;
        request.id = id;
        request.prompt = "<start_of_turn>user\nTell me about moths.<end_of_turn>\n<start_of_turn>model\n";
        request.gen_config.seed = 42;
        request.gen_config.temperature = 0.0f;
        request.gen_config.max_tokens = max_tokens;
        request.gen_config.stop = stops;
        request.on_event = [result, mutex, cv](const Chorus::ChorusSignal& signal) {
            std::lock_guard<std::mutex> lock(*mutex);
            if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event))
                result->chunks.push_back(std::get<Chorus::ChorusSignal::Token>(signal.event).text);
            else if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event))
                result->stopped = true;
            else if (std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
                result->errored = true;
            if (result->stopped || result->errored)
                cv->notify_one();
        };
        engine.submit_request(request);

        std::unique_lock<std::mutex> lock(*mutex);
        result->timed_out =
            !cv->wait_for(lock, std::chrono::seconds(15), [&] { return result->stopped || result->errored; });
        return result;
    };

    const auto baseline = run(211, {}, 24);
    ASSERT_TRUE(!baseline->timed_out);
    ASSERT_TRUE(baseline->stopped);
    ASSERT_TRUE(!baseline->errored);
    std::string baseline_text;
    for (const auto& chunk : baseline->chunks)
        baseline_text += chunk;
    ASSERT_TRUE(baseline_text.size() >= 4);

    const size_t marker_position = baseline_text.size() / 2;
    const std::string marker = baseline_text.substr(marker_position);
    const auto filtered = run(212, {marker}, 24);
    ASSERT_TRUE(!filtered->timed_out);
    ASSERT_TRUE(filtered->stopped);
    ASSERT_TRUE(!filtered->errored);
    std::string filtered_text;
    for (const auto& chunk : filtered->chunks) {
        ASSERT_TRUE(chunk.find(marker) == std::string::npos);
        filtered_text += chunk;
    }
    ASSERT_TRUE(filtered_text.find(marker) == std::string::npos);
    ASSERT_EQ(filtered_text, baseline_text.substr(0, marker_position));

    const auto unmatched = run(213, {baseline_text + "<UNMATCHED>"}, 24);
    ASSERT_TRUE(!unmatched->timed_out);
    ASSERT_TRUE(unmatched->stopped);
    ASSERT_TRUE(!unmatched->errored);
    std::string unmatched_text;
    for (const auto& chunk : unmatched->chunks)
        unmatched_text += chunk;
    ASSERT_EQ(unmatched_text, baseline_text);

    const auto reused = run(214, {}, 2);
    ASSERT_TRUE(!reused->timed_out);
    ASSERT_TRUE(reused->stopped);
    ASSERT_TRUE(!reused->errored);
    ASSERT_TRUE(!reused->chunks.empty());
    engine.shutdown();
}

namespace {

enum class ReentryTrigger { ZeroBudget, Rejection };

constexpr std::string_view REENTRY_ZERO_BUDGET_CHILD = "llama_reentry_zero_budget";
constexpr std::string_view REENTRY_REJECTION_CHILD = "llama_reentry_rejection";

struct ReentryState {
    std::mutex mutex;
    std::condition_variable cv;
    bool trigger_completed = false;
    bool followup_stopped = false;
    Chorus::LlamaEngine* engine = nullptr;
};

int run_reentry_child(ReentryTrigger trigger) {
    auto state = std::make_shared<ReentryState>();
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    Chorus::LlamaEngine engine;
    state->engine = &engine;
    if (engine.initialize(config, {}, {}).has_value())
        return 10;

    Chorus::ChorusRequest request;
    request.id = 221;
    if (trigger == ReentryTrigger::ZeroBudget)
        request.gen_config.max_tokens = 0;
    else
        request.gen_config.max_tokens = -2;
    request.on_event = [state, trigger](const Chorus::ChorusSignal& signal) {
        const bool expected = trigger == ReentryTrigger::ZeroBudget
                                  ? std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event)
                                  : std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
        if (!expected)
            return;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->trigger_completed = true;
        }

        Chorus::ChorusRequest followup;
        followup.id = 222;
        followup.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
        followup.gen_config.max_tokens = 2;
        followup.on_event = [state](const Chorus::ChorusSignal& next_signal) {
            if (std::holds_alternative<Chorus::ChorusSignal::Stop>(next_signal.event) ||
                std::holds_alternative<Chorus::ChorusSignal::Error>(next_signal.event)) {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->followup_stopped = std::holds_alternative<Chorus::ChorusSignal::Stop>(next_signal.event);
                state->cv.notify_one();
            }
        };
        state->engine->submit_request(followup);
    };
    engine.submit_request(request);

    bool completed = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        completed = state->cv.wait_for(lock, std::chrono::seconds(5), [&] { return state->followup_stopped; });
    }
    engine.shutdown();
    return completed && state->trigger_completed ? 0 : 11;
}

bool run_reentry_isolated(ReentryTrigger trigger) {
    const std::string_view child_name =
        trigger == ReentryTrigger::ZeroBudget ? REENTRY_ZERO_BUDGET_CHILD : REENTRY_REJECTION_CHILD;
    return run_isolated_test_child(std::string(child_name), std::chrono::seconds(15));
}

} // namespace

TEST_F(LlamaIntegrationModelTest, Llama_stop_zero_callback_can_submit_followup) {
    ASSERT_TRUE(run_reentry_isolated(ReentryTrigger::ZeroBudget));
}

TEST_F(LlamaIntegrationModelTest, Llama_stop_rejection_callback_can_submit_followup) {
    ASSERT_TRUE(run_reentry_isolated(ReentryTrigger::Rejection));
}

int run_llama_reentry_child_mode(std::string_view child_name) {
    if (child_name == REENTRY_ZERO_BUDGET_CHILD)
        return run_reentry_child(ReentryTrigger::ZeroBudget);
    if (child_name == REENTRY_REJECTION_CHILD)
        return run_reentry_child(ReentryTrigger::Rejection);
    return 64;
}

TEST_F(LlamaIntegrationModelTest, Llama_stop_completion_releases_callback_resources) {

    struct State {
        std::atomic<bool> terminal{false};
    };
    auto state = std::make_shared<State>();
    auto owned = std::make_shared<int>(42);
    std::weak_ptr<int> ownership_probe = owned;

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest request;
    request.id = 223;
    request.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    request.gen_config.max_tokens = 1;
    request.on_event = [owned, state](const Chorus::ChorusSignal& signal) {
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            state->terminal = true;
    };
    engine.submit_request(request);
    request.on_event = {};
    owned.reset();

    int timeout_ms = 5000;
    while ((!state->terminal || !ownership_probe.expired()) && timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timeout_ms -= 10;
    }
    ASSERT_TRUE(state->terminal);
    ASSERT_TRUE(ownership_probe.expired());
    engine.shutdown();
}

TEST_F(LlamaIntegrationModelTest, Llama_stop_zero_completes_while_slot_is_occupied) {

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool busy_started = false;
        bool busy_terminal = false;
        std::vector<Chorus::ChorusSignal::Event> zero_events;
        bool busy_terminal_at_zero = false;
    };
    auto state = std::make_shared<State>();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest busy;
    busy.id = 224;
    busy.prompt = "<start_of_turn>user\nTell me about moths.<end_of_turn>\n<start_of_turn>model\n";
    busy.gen_config.max_tokens = 512;
    busy.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    busy.on_event = [state](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            state->busy_started = true;
            state->cv.notify_one();
        } else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
        )
            state->busy_terminal = true;
    };
    engine.submit_request(busy);

    {
        std::unique_lock<std::mutex> lock(state->mutex);
        ASSERT_TRUE(state->cv.wait_for(lock, std::chrono::seconds(15), [&] { return state->busy_started; }));
    }

    Chorus::ChorusRequest zero;
    zero.id = 225;
    zero.gen_config.max_tokens = 0;
    zero.on_event = [state](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->zero_events.push_back(signal.event);
        state->busy_terminal_at_zero = state->busy_terminal;
        state->cv.notify_one();
    };
    engine.submit_request(zero);

    std::unique_lock<std::mutex> lock(state->mutex);
    ASSERT_TRUE(state->cv.wait_for(lock, std::chrono::seconds(1), [&] { return !state->zero_events.empty(); }));
    ASSERT_EQ(state->zero_events.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state->zero_events[0]));
    ASSERT_TRUE(!state->busy_terminal_at_zero);
    lock.unlock();
    engine.shutdown();
}

// Terminal invariant sweep: every accepted request yields exactly one terminal
// event. One single-request engine drives success, token limit, zero limit,
// stop match, cancellation, and invalid-grammar-after-acceptance in sequence; the
// remaining scenarios (decode failure, engine stop) need their own engine
// lifecycle and follow below.
TEST_F(LlamaIntegrationModelTest, Llama_terminal_invariant_one_per_request) {

    struct Sweep {
        std::mutex mutex;
        std::condition_variable cv;
        std::map<int64_t, std::vector<Chorus::ChorusSignal>> terminals;
        std::map<int64_t, std::string> text;
        std::map<int64_t, size_t> tokens;
    };
    auto sweep = std::make_shared<Sweep>();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    auto drive = [&](Chorus::ChorusRequest request, bool cancel_on_first_token) -> bool {
        const int64_t id = request.id;
        request.on_event = [sweep, &engine, id, cancel_on_first_token](const Chorus::ChorusSignal& signal) {
            bool do_cancel = false;
            {
                std::lock_guard<std::mutex> lock(sweep->mutex);
                if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
                    sweep->text[id] += std::get<Chorus::ChorusSignal::Token>(signal.event).text;
                    if (++sweep->tokens[id] == 1 && cancel_on_first_token)
                        do_cancel = true;
                } else if (
                    std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
                    std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
                ) {
                    sweep->terminals[id].push_back(signal);
                }
            }
            if (do_cancel)
                engine.cancel_request(id);
            sweep->cv.notify_all();
        };
        engine.submit_request(request);
        std::unique_lock<std::mutex> lock(sweep->mutex);
        return sweep->cv.wait_for(lock, std::chrono::seconds(15), [&] { return !sweep->terminals[id].empty(); });
    };

    const std::string moth_prompt = "<start_of_turn>user\nTell me about moths.<end_of_turn>\n<start_of_turn>model\n";

    // success: a plain bounded completion terminates once with Stop.
    Chorus::ChorusRequest success;
    success.id = 401;
    success.prompt = moth_prompt;
    success.gen_config.seed = 42;
    success.gen_config.temperature = 0.0f;
    success.gen_config.max_tokens = 16;
    if (!drive(success, false)) {
        engine.shutdown();
        ASSERT_TRUE(false);
    }
    const std::string success_text = sweep->text[401];
    if (success_text.size() < 4) {
        engine.shutdown();
        ASSERT_TRUE(success_text.size() >= 4);
    }

    // token limit: a long prompt with a small budget still terminates once.
    Chorus::ChorusRequest limited;
    limited.id = 402;
    {
        std::string prompt = "<start_of_turn>user\n";
        for (int i = 0; i < 8; ++i)
            prompt += "Tell me a long story about dragons. ";
        prompt += "<end_of_turn>\n<start_of_turn>model\n";
        limited.prompt = prompt;
    }
    limited.gen_config.max_tokens = 4;
    if (!drive(limited, false)) {
        engine.shutdown();
        ASSERT_TRUE(false);
    }

    // zero limit: completes without entering inference, one Stop.
    Chorus::ChorusRequest zero;
    zero.id = 403;
    zero.prompt = "This prompt must not enter inference.";
    zero.gen_config.max_tokens = 0;
    if (!drive(zero, false)) {
        engine.shutdown();
        ASSERT_TRUE(false);
    }

    // stop match: identical deterministic generation truncated by a marker suffix.
    const std::string marker = success_text.substr(success_text.size() / 2);
    Chorus::ChorusRequest stop_match;
    stop_match.id = 404;
    stop_match.prompt = moth_prompt;
    stop_match.gen_config.seed = 42;
    stop_match.gen_config.temperature = 0.0f;
    stop_match.gen_config.max_tokens = 16;
    stop_match.gen_config.stop = {marker};
    if (!drive(stop_match, false)) {
        engine.shutdown();
        ASSERT_TRUE(false);
    }

    // cancellation: a long request cancelled after its first token ends once with Cancelled.
    Chorus::ChorusRequest cancelled;
    cancelled.id = 405;
    cancelled.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
    cancelled.gen_config.max_tokens = 512;
    cancelled.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    if (!drive(cancelled, true)) {
        engine.shutdown();
        ASSERT_TRUE(false);
    }

    // invalid grammar after acceptance: a well-formed request whose grammar fails at
    // sampler construction ends once with an InvalidRequest error.
    Chorus::ChorusRequest bad_grammar;
    bad_grammar.id = 406;
    bad_grammar.prompt = moth_prompt;
    bad_grammar.gen_config.max_tokens = 8;
    bad_grammar.gen_config.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= ["};
    if (!drive(bad_grammar, false)) {
        engine.shutdown();
        ASSERT_TRUE(false);
    }

    engine.shutdown();

    // Exactly one terminal per accepted request, of the expected kind.
    ASSERT_EQ(sweep->terminals[401].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(sweep->terminals[401][0].event));
    ASSERT_EQ(sweep->terminals[402].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(sweep->terminals[402][0].event));
    ASSERT_EQ(sweep->terminals[403].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(sweep->terminals[403][0].event));
    ASSERT_EQ(sweep->terminals[404].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(sweep->terminals[404][0].event));
    ASSERT_EQ(sweep->terminals[405].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(sweep->terminals[405][0].event));
    ASSERT_TRUE(
        std::get<Chorus::ChorusSignal::Error>(sweep->terminals[405][0].event).code == Chorus::ChorusError::Cancelled
    );
    ASSERT_EQ(sweep->terminals[406].size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(sweep->terminals[406][0].event));
    ASSERT_TRUE(
        std::get<Chorus::ChorusSignal::Error>(sweep->terminals[406][0].event).code ==
        Chorus::ChorusError::InvalidRequest
    );
}

// Terminal invariant: a runtime decode failure ends the request once with an Error.
TEST_F(LlamaIntegrationModelTest, Llama_terminal_invariant_decode_failure_ends_once) {

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::ChorusSignal> terminals;

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false},
        {"context_size", int64_t{64}},
        {"max_concurrent_requests", int64_t{1}},
    };
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    std::string huge_prompt;
    for (int i = 0; i < 200; ++i)
        huge_prompt += "The quick brown fox jumps over the lazy dog. ";

    Chorus::ChorusRequest request;
    request.id = 411;
    request.prompt = huge_prompt;
    request.gen_config.max_tokens = 8;
    request.on_event = [&](const Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::lock_guard<std::mutex> lock(mutex);
        terminals.push_back(signal);
        cv.notify_one();
    };
    engine.submit_request(request);

    bool terminal = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        terminal = cv.wait_for(lock, std::chrono::seconds(15), [&] { return !terminals.empty(); });
    }
    engine.shutdown();

    ASSERT_TRUE(terminal);
    ASSERT_EQ(terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(terminals[0].event));
    ASSERT_TRUE(std::get<Chorus::ChorusSignal::Error>(terminals[0].event).code == Chorus::ChorusError::Decode);
}

// Terminal invariant: stopping the engine mid-flight drains the active request with
// exactly one Cancelled terminal.
TEST_F(LlamaIntegrationModelTest, Llama_terminal_invariant_engine_shutdown_ends_once) {

    std::mutex mutex;
    std::condition_variable cv;
    bool started = false;
    std::vector<Chorus::ChorusSignal> terminals;

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{1}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest request;
    request.id = 421;
    request.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
    request.gen_config.max_tokens = 512;
    request.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    request.on_event = [&](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            started = true;
            cv.notify_all();
        } else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
        ) {
            terminals.push_back(signal);
            cv.notify_all();
        }
    };
    engine.submit_request(request);

    bool active = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        active = cv.wait_for(lock, std::chrono::seconds(15), [&] { return started; });
    }
    if (!active) {
        engine.shutdown();
        ASSERT_TRUE(active);
    }

    engine.shutdown(); // tears down mid-flight; must synthesize exactly one terminal

    ASSERT_EQ(terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(terminals[0].event));
    ASSERT_TRUE(std::get<Chorus::ChorusSignal::Error>(terminals[0].event).code == Chorus::ChorusError::Cancelled);
    ASSERT_EQ(terminals[0].request_id, request.id);
}

// Two-sequence scenario: one request is cancelled while the other completes
// normally, each ending with exactly one terminal of its own kind.
TEST_F(LlamaIntegrationModelTest, Llama_two_sequences_one_cancels_one_completes) {

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool cancel_started = false;
        std::vector<Chorus::ChorusSignal> cancel_terminals;
        std::vector<Chorus::ChorusSignal> complete_terminals;
    };
    auto state = std::make_shared<State>();

    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"max_concurrent_requests", int64_t{2}}};
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    const int64_t cancel_id = 431;
    const int64_t complete_id = 432;

    Chorus::ChorusRequest cancel_req;
    cancel_req.id = cancel_id;
    cancel_req.prompt = "<start_of_turn>user\nTell me a very long story.<end_of_turn>\n<start_of_turn>model\n";
    cancel_req.gen_config.max_tokens = 512;
    cancel_req.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"ignore_eos", true}};
    cancel_req.on_event = [state](const Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            state->cancel_started = true;
            state->cv.notify_all();
        } else if (
            std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
            std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)
        ) {
            state->cancel_terminals.push_back(signal);
            state->cv.notify_all();
        }
    };

    Chorus::ChorusRequest complete_req;
    complete_req.id = complete_id;
    complete_req.prompt = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n";
    complete_req.gen_config.max_tokens = 8;
    complete_req.on_event = [state](const Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) &&
            !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        state->complete_terminals.push_back(signal);
        state->cv.notify_all();
    };

    engine.submit_request(cancel_req);
    engine.submit_request(complete_req);

    bool started = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        started = state->cv.wait_for(lock, std::chrono::seconds(15), [&] { return state->cancel_started; });
    }
    if (!started) {
        engine.shutdown();
        ASSERT_TRUE(started);
    }

    engine.cancel_request(cancel_id);

    bool both = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        both = state->cv.wait_for(lock, std::chrono::seconds(20), [&] {
            return !state->cancel_terminals.empty() && !state->complete_terminals.empty();
        });
    }
    engine.shutdown();

    ASSERT_TRUE(both);
    ASSERT_EQ(state->cancel_terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(state->cancel_terminals[0].event));
    ASSERT_TRUE(
        std::get<Chorus::ChorusSignal::Error>(state->cancel_terminals[0].event).code == Chorus::ChorusError::Cancelled
    );
    ASSERT_EQ(state->complete_terminals.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(state->complete_terminals[0].event));
}

// Chat support using the real model.

namespace {

Chorus::ChorusConfig make_chat_config() {
    Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}};
    return config;
}

struct RuntimeDrainResult {
    std::optional<Chorus::RuntimeEvent::Kind> terminal_kind;
    bool saw_truncation = false;
    std::string complete_text;
};

// Polls the runtime until a Complete/Error terminal or timeout, recording the
// Complete text and whether a HistoryTruncated event fired en route.
RuntimeDrainResult drain_runtime_until_terminal(Chorus::ChorusRuntime& runtime, int timeout_ms = 60000) {
    RuntimeDrainResult result;
    while (timeout_ms > 0) {
        for (auto& event : runtime.poll()) {
            if (event.kind == Chorus::RuntimeEvent::Kind::HistoryTruncated)
                result.saw_truncation = true;
            if (event.kind == Chorus::RuntimeEvent::Kind::Complete) {
                result.complete_text = event.text;
                result.terminal_kind = event.kind;
            }
            if (event.kind == Chorus::RuntimeEvent::Kind::Error)
                result.terminal_kind = event.kind;
        }
        if (result.terminal_kind)
            return result;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        timeout_ms -= 50;
    }
    return result;
}

std::optional<std::vector<float>> drain_embedding(Chorus::ChorusRuntime& runtime, Chorus::RequestId request_id) {
    for (int waited_ms = 0; waited_ms < 60000; waited_ms += 50) {
        for (auto& event : runtime.poll()) {
            if (event.request_id != request_id)
                continue;
            if (event.kind == Chorus::RuntimeEvent::Kind::Embedding)
                return std::move(event.embedding);
            if (event.kind == Chorus::RuntimeEvent::Kind::Error)
                return std::nullopt;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return std::nullopt;
}

} // namespace

TEST_F(LlamaIntegrationModelTest, Llama_embeddinggemma_returns_a_normalized_embedding) {
    Chorus::ChorusConfig config = make_gguf_config("tests/models/embeddinggemma-300M-Q8_0.gguf");
    config.provider_options["llama"] =
        Chorus::ProviderOptionMap{{"use_gpu", false}, {"n_batch", int64_t{64}}, {"n_ubatch", int64_t{64}}};
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(load_runtime(runtime, Chorus::make_engine(Chorus::Provider::Llama), config).ok());
    const auto capabilities = runtime.capabilities();
    ASSERT_TRUE(capabilities.has_value() && capabilities->embeddings);

    const auto submitted = runtime.submit(Chorus::EmbeddingRequest{"The cat sat on the mat.", std::nullopt, 0});
    ASSERT_TRUE(submitted.ok());
    std::optional<Chorus::RuntimeEvent> terminal;
    for (int waited_ms = 0; waited_ms < 60000 && !terminal; waited_ms += 50) {
        for (auto& event : runtime.poll()) {
            if (event.request_id == submitted.request_id && (event.kind == Chorus::RuntimeEvent::Kind::Embedding ||
                                                             event.kind == Chorus::RuntimeEvent::Kind::Error))
                terminal = std::move(event);
        }
        if (!terminal)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(terminal.has_value());
    ASSERT_TRUE(terminal->kind == Chorus::RuntimeEvent::Kind::Embedding);
    ASSERT_EQ(terminal->embedding.size(), size_t{768});
    double norm = 0.0;
    for (float value : terminal->embedding) {
        ASSERT_TRUE(std::isfinite(value));
        norm += static_cast<double>(value) * value;
    }
    ASSERT_NEAR(norm, 1.0, 1e-5);

    std::string oversized_prompt;
    for (int i = 0; i < 200; ++i)
        oversized_prompt += "word ";
    const auto oversized = runtime.submit(Chorus::EmbeddingRequest{oversized_prompt, std::nullopt, 0});
    ASSERT_TRUE(oversized.ok());
    const auto oversized_events = drain_runtime_events(runtime);
    ASSERT_EQ(oversized_events.back().request_id, oversized.request_id);
    ASSERT_EQ(oversized_events.back().kind, Chorus::RuntimeEvent::Kind::Error);
    ASSERT_EQ(oversized_events.back().error, Chorus::ChorusError::InvalidRequest);
}

TEST_F(LlamaIntegrationModelTest, Llama_embeddinggemma_none_pooling_uses_normalized_token_embeddings) {
    Chorus::ChorusConfig config = make_gguf_config("tests/models/embeddinggemma-300M-Q8_0.gguf");
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"use_gpu", false}, {"n_batch", int64_t{64}}, {"n_ubatch", int64_t{64}}, {"pooling", std::string{"none"}}
    };
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(load_runtime(runtime, Chorus::make_engine(Chorus::Provider::Llama), config).ok());

    const auto submit = [&](const std::string& prompt) {
        const auto result = runtime.submit(Chorus::EmbeddingRequest{prompt, std::nullopt, 0});
        if (!result.ok())
            return std::optional<std::vector<float>>{};
        return drain_embedding(runtime, result.request_id);
    };
    const auto anchor = submit("The cat sat on the mat.");
    const auto paraphrase = submit("A feline was resting on the rug.");
    const auto unrelated = submit("Volcanic eruptions produce molten rock.");
    ASSERT_TRUE(anchor.has_value());
    ASSERT_TRUE(paraphrase.has_value());
    ASSERT_TRUE(unrelated.has_value());
    ASSERT_EQ(anchor->size(), size_t{768});

    double norm = 0.0;
    double related_score = 0.0;
    double unrelated_score = 0.0;
    for (size_t i = 0; i < anchor->size(); ++i) {
        ASSERT_TRUE(std::isfinite((*anchor)[i]));
        norm += static_cast<double>((*anchor)[i]) * (*anchor)[i];
        related_score += static_cast<double>((*anchor)[i]) * (*paraphrase)[i];
        unrelated_score += static_cast<double>((*anchor)[i]) * (*unrelated)[i];
    }
    ASSERT_NEAR(norm, 1.0, 1e-5);
    ASSERT_TRUE(related_score > unrelated_score);
}

TEST_F(LlamaIntegrationModelTest, A_generation_model_serves_embeddings_only_once_they_are_turned_on) {
    const auto embed_with = [](bool embeddings) {
        Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
        config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}, {"embeddings", embeddings}};
        Chorus::ChorusRuntime runtime;
        EXPECT_TRUE(load_runtime(runtime, Chorus::make_engine(Chorus::Provider::Llama), config).ok());
        const auto result = runtime.submit(Chorus::EmbeddingRequest{"The cat sat on the mat.", std::nullopt, 0});
        if (!result.ok())
            return std::variant<Chorus::ChorusError, std::vector<float>>{result.error};
        return std::variant<Chorus::ChorusError, std::vector<float>>{
            drain_embedding(runtime, result.request_id).value_or(std::vector<float>{})
        };
    };

    const auto off = embed_with(false);
    const auto on = embed_with(true);

    ASSERT_EQ(std::get<Chorus::ChorusError>(off), Chorus::ChorusError::UnsupportedFeature);
    ASSERT_EQ(std::get<std::vector<float>>(on).size(), size_t{640});
}

TEST_F(LlamaIntegrationModelTest, A_generation_model_without_embeddings_reserves_a_fraction_of_the_compute_memory) {
    const auto compute_mib = [](bool embeddings) {
        EngineLogCapture log_capture;
        Chorus::ChorusConfig config = make_gguf_config(MODEL_PATH);
        config.provider_options["llama"] = Chorus::ProviderOptionMap{{"use_gpu", false}, {"embeddings", embeddings}};
        Chorus::LlamaEngine engine;
        EXPECT_TRUE(!engine.initialize(config, log_capture.logger(), {}).has_value());
        engine.shutdown();
        const std::string logs = log_capture.text();
        const std::string marker = "CPU compute buffer size =";
        const size_t found = logs.find(marker);
        return found == std::string::npos ? -1.0 : std::stod(logs.substr(found + marker.size()));
    };

    const double without = compute_mib(false);
    const double with = compute_mib(true);

    ASSERT_GT(without, 0.0);
    ASSERT_LT(without * 4, with);
}

TEST_F(LlamaIntegrationModelTest, Llama_chat_messages_render_and_generate) {
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_chat_config(), {}, {}).has_value());

    // Render hook: gemma's embedded template must produce its role scaffolding.
    auto rendered = engine.render_chat_prompt(
        {{Chorus::MessageRole::System, Chorus::MessageContent::text("You are terse.")},
         {Chorus::MessageRole::User, Chorus::MessageContent::text("Say hi.")}},
        std::nullopt,
        true
    );
    ASSERT_TRUE(rendered.has_value());
    ASSERT_TRUE(rendered->text.find("<start_of_turn>user") != std::string::npos);
    ASSERT_TRUE(rendered->token_count > 0);

    // A custom override changes the rendering.
    auto overridden = engine.render_chat_prompt(
        {{Chorus::MessageRole::User, Chorus::MessageContent::text("Say hi.")}},
        "{%- for m in messages -%}[[{{ m.role }}]]{{ m.content }}{%- endfor -%}",
        true
    );
    ASSERT_TRUE(overridden.has_value());
    ASSERT_TRUE(overridden->text.find("[[user]]") != std::string::npos);

    // Messages-carrying request generates a completion.
    std::mutex mutex;
    std::string text;
    std::atomic<bool> done{false};
    std::atomic<bool> stopped{false};
    Chorus::ChorusRequest request;
    request.id = 901;
    request.messages = {{Chorus::MessageRole::User, Chorus::MessageContent::text("Reply with the single word: hello")}};
    request.gen_config.max_tokens = 16;
    request.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event))
            text += std::get<Chorus::ChorusSignal::Token>(sig.event).text;
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event)) {
            stopped = true;
            done = true;
        } else if (std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
            done = true;
        }
    };
    engine.submit_request(request);
    for (int waited_ms = 0; waited_ms < 30000 && !done; waited_ms += 50)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_TRUE(done.load());
    ASSERT_TRUE(stopped.load());
    ASSERT_TRUE(!text.empty());

    request.id = 902;
    request.chat_template = "{%- for m in messages -%}[[{{ m.role }}]]{{ m.content }}{%- endfor -%}[[assistant]]";
    request.gen_config.show_thinking = false;
    request.gen_config.max_tokens = 1;
    done = false;
    stopped = false;
    text.clear();
    engine.submit_request(request);
    for (int waited_ms = 0; waited_ms < 30000 && !done; waited_ms += 50)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_TRUE(done.load());
    ASSERT_TRUE(stopped.load());
    engine.shutdown();
}

TEST_F(LlamaIntegrationModelTest, Llama_chat_capabilities_report_rendering) {
    Chorus::LlamaEngine engine;
    ASSERT_TRUE(!engine.initialize(make_chat_config(), {}, {}).has_value());
    ASSERT_TRUE(engine.capabilities().prompt_rendering);
    auto info = engine.loaded_model_info();
    ASSERT_TRUE(info.has_value() && info->per_request_context.has_value());
    ASSERT_TRUE(*info->per_request_context > 0);
    engine.shutdown();
}

TEST_F(LlamaIntegrationModelTest, Llama_chat_multi_turn_stays_contextual) {
    // Exercise a multi-turn conversation at the runtime layer so history
    // assembly itself is under test.
    Chorus::ChorusRuntime runtime;
    ASSERT_TRUE(load_runtime(runtime, Chorus::make_engine(Chorus::Provider::Llama), make_chat_config()).ok());

    Chorus::GenerationRequest turn1;
    turn1.prompt = "My name is Trebor. Remember my name.";
    turn1.session_id = "npc_1";
    turn1.options.max_tokens = 48;
    turn1.options.temperature = 0.0f; // greedy: deterministic recall
    ASSERT_TRUE(runtime.submit(turn1).ok());
    ASSERT_TRUE(drain_runtime_until_terminal(runtime).terminal_kind == Chorus::RuntimeEvent::Kind::Complete);

    Chorus::GenerationRequest turn2;
    turn2.prompt = "What is my name? Answer with just the name.";
    turn2.session_id = "npc_1";
    turn2.options.max_tokens = 24;
    turn2.options.temperature = 0.0f;
    ASSERT_TRUE(runtime.submit(turn2).ok());
    auto drained = drain_runtime_until_terminal(runtime);
    ASSERT_TRUE(drained.terminal_kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_TRUE(drained.complete_text.find("Trebor") != std::string::npos);
}

TEST_F(LlamaIntegrationModelTest, Llama_chat_truncation_preserves_system) {
    // A tiny context (max_concurrent_requests=1) forces a real truncation, then the
    // render hook proves the system message survived in the fitted window.
    Chorus::ChorusRuntime runtime;
    auto config = make_gguf_config(MODEL_PATH);
    config.provider_options["llama"] = Chorus::ProviderOptionMap{
        {"context_size", int64_t{512}}, {"max_concurrent_requests", int64_t{1}}, {"use_gpu", false}
    };
    ASSERT_TRUE(load_runtime(runtime, Chorus::make_engine(Chorus::Provider::Llama), config).ok());

    std::vector<Chorus::ConversationMessage> history{
        {0, {Chorus::MessageRole::System, Chorus::MessageContent::text("You are Brunn the blacksmith.")}}
    };
    for (int i = 0; i < 30; ++i) {
        history.push_back(
            {static_cast<Chorus::MessageId>(1 + i * 2),
             {Chorus::MessageRole::User,
              Chorus::MessageContent::text("Filler question number " + std::to_string(i) + " about the weather.")}}
        );
        history.push_back(
            {static_cast<Chorus::MessageId>(2 + i * 2),
             {Chorus::MessageRole::Assistant,
              Chorus::MessageContent::text("A filler answer about the weather, number " + std::to_string(i) + ".")}}
        );
    }
    ASSERT_TRUE(!runtime.import_conversation_history("npc_1", std::move(history)).has_value());

    Chorus::GenerationConfig gen;
    gen.max_tokens = 64;
    Chorus::GenerationRequest preview;
    preview.session_id = "npc_1";
    preview.prompt = "Who are you?";
    preview.options = gen;
    auto fitted = runtime.render_prompt(preview);
    ASSERT_TRUE(fitted.ok());
    const auto fitted_events = drain_runtime_events(runtime);
    ASSERT_EQ(fitted_events.back().kind, Chorus::RuntimeEvent::Kind::PromptRendered);
    ASSERT_TRUE(fitted_events.back().text.find("Brunn the blacksmith") != std::string::npos); // system pinned
    ASSERT_TRUE(fitted_events.back().text.find("number 0 ") == std::string::npos);            // oldest dropped

    Chorus::GenerationRequest turn;
    turn.prompt = "Who are you?";
    turn.session_id = "npc_1";
    turn.options = gen;
    ASSERT_TRUE(runtime.submit(turn).ok());
    auto drained = drain_runtime_until_terminal(runtime);
    ASSERT_TRUE(drained.terminal_kind == Chorus::RuntimeEvent::Kind::Complete);
    ASSERT_TRUE(drained.saw_truncation);
}
