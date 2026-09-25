#include "chorus/core/common.hpp"
#include "chorus/providers/echo/echo_engine.hpp"
#include "collecting_log.hpp"
#include "engine_contract_suite.hpp"
#include "process_test.hpp"
#include "gtest_utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

TEST(EchoEngine, Echo_initializes_without_a_model_file) {
    Chorus::EchoEngine engine;
    ASSERT_TRUE(!engine.is_initialized());

    Chorus::ChorusConfig config; // empty InitialModelSpec deliberately: Echo needs no model
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.shutdown();
    ASSERT_TRUE(!engine.is_initialized());
}

TEST(EchoEngine, Echo_progress_cancellation_leaves_no_worker_and_allows_reload) {
    Chorus::EchoEngine engine;
    std::stop_source stop;
    std::vector<Chorus::LoadProgress> progress;
    Chorus::InitializationControl control{stop.get_token(), [&](const Chorus::LoadProgress& sample) {
        progress.push_back(sample);
        stop.request_stop();
    }};
    const auto failure = engine.initialize({}, {}, control);
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(failure->error, Chorus::ChorusError::Cancelled);
    EXPECT_FALSE(engine.is_initialized());
    ASSERT_EQ(progress.size(), size_t{1});
    EXPECT_EQ(progress.front().phase, Chorus::LoadPhase::InitializingEngine);
    EXPECT_FALSE(progress.front().fraction.has_value());
    ASSERT_FALSE(engine.initialize({}, {}, {}).has_value());
    ASSERT_TRUE(engine.is_initialized());
    engine.shutdown();
}

TEST(EchoEngine, Echo_streams_prompt_word_by_word_then_stops) {
    std::mutex sig_mutex;
    std::vector<Chorus::ChorusSignal> sigs;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest req;
    req.id = 7;
    req.prompt = "hello chorus seam";
    req.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(sig_mutex);
        sigs.push_back(sig);
    };
    engine.submit_request(req);

    int timeout_ms = 2000;
    while (timeout_ms > 0) {
        {
            std::lock_guard<std::mutex> lock(sig_mutex);
            if (!sigs.empty() && std::holds_alternative<Chorus::ChorusSignal::Stop>(sigs.back().event))
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timeout_ms -= 10;
    }

    std::lock_guard<std::mutex> lock(sig_mutex);
    // 3 words -> 3 Token signals + 1 Stop, all carrying the request id.
    ASSERT_EQ(sigs.size(), 4);
    std::string reassembled;
    for (size_t i = 0; i + 1 < sigs.size(); ++i) {
        ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(sigs[i].event));
        ASSERT_EQ(sigs[i].request_id, 7);
        reassembled += std::get<Chorus::ChorusSignal::Token>(sigs[i].event).text;
    }
    ASSERT_TRUE(reassembled == "hello chorus seam");
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(sigs.back().event));
    ASSERT_EQ(sigs.back().request_id, 7);
    const size_t terminals = std::count_if(sigs.begin(), sigs.end(), [](const auto& signal) {
        return std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
               std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
    });
    ASSERT_EQ(terminals, size_t{1});

    engine.shutdown();
}

TEST(EchoEngine, Echo_embedding_is_deterministic_normalized_and_lexical) {
    const auto embed = [](const std::string& prompt) {
        Chorus::EchoEngine engine;
        EXPECT_TRUE(!engine.initialize(Chorus::ChorusConfig{}, {}, {}).has_value());
        Chorus::ChorusRequest request;
        request.id = 1;
        request.type = Chorus::RequestType::Embedding;
        request.prompt = prompt;
        EXPECT_TRUE(!engine.validate_request(request).has_value());

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> signals;
        request.on_event = [&](Chorus::ChorusSignal& signal) {
            std::lock_guard<std::mutex> lock(mutex);
            signals.push_back(signal);
            cv.notify_all();
        };
        engine.submit_request(request);
        {
            std::unique_lock<std::mutex> lock(mutex);
            EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] {
                return !signals.empty() && std::holds_alternative<Chorus::ChorusSignal::Stop>(signals.back().event);
            }));
        }
        engine.shutdown();
        EXPECT_EQ(signals.size(), size_t{2});
        if (signals.empty() || !std::holds_alternative<Chorus::ChorusSignal::Embedding>(signals[0].event)) {
            ADD_FAILURE() << "Echo did not emit an embedding before Stop.";
            return std::vector<float>{};
        }
        return std::get<Chorus::ChorusSignal::Embedding>(signals[0].event).values;
    };
    const auto anchor = embed("A cat sleeps on the warm mat");
    const auto same_case_folded = embed("a CAT sleeps on the warm mat");
    const auto related = embed("cat sleeps on a mat");
    const auto unrelated = embed("volcanic eruptions and molten rock");

    ASSERT_EQ(anchor.size(), size_t{128});
    ASSERT_EQ(anchor, same_case_folded);
    double norm = 0.0;
    double related_score = 0.0;
    double unrelated_score = 0.0;
    for (size_t i = 0; i < anchor.size(); ++i) {
        ASSERT_TRUE(std::isfinite(anchor[i]));
        norm += static_cast<double>(anchor[i]) * anchor[i];
        related_score += static_cast<double>(anchor[i]) * related[i];
        unrelated_score += static_cast<double>(anchor[i]) * unrelated[i];
    }
    ASSERT_NEAR(norm, 1.0, 1e-6);
    ASSERT_TRUE(related_score > unrelated_score);
}

TEST(EchoEngine, Echo_embedding_rejects_empty_prompt) {
    Chorus::EchoEngine engine;
    ASSERT_TRUE(!engine.initialize(Chorus::ChorusConfig{}, {}, {}).has_value());
    Chorus::ChorusRequest request;
    request.type = Chorus::RequestType::Embedding;
    const auto rejection = engine.validate_request(request);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::InvalidRequest);
    engine.shutdown();
}

TEST(EchoEngine, Echo_capabilities_deterministic_across_init) {
    Chorus::EchoEngine engine;
    auto before = engine.capabilities();
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());
    auto after = engine.capabilities();
    ASSERT_EQ(before.provider_id, std::string("echo"));
    ASSERT_EQ(after.provider_id, before.provider_id);
    ASSERT_TRUE(before.streaming && after.streaming);
    ASSERT_TRUE(before.scheduling == Chorus::SchedulingAuthority::ProviderManaged);
    ASSERT_TRUE(before.cancellation && after.cancellation);
    const std::set<std::string> advertised_common(
        before.common_generation_options.begin(), before.common_generation_options.end()
    );
    ASSERT_TRUE(advertised_common == std::set<std::string>{"max_tokens"});
    ASSERT_TRUE(
        std::set<std::string>(after.common_generation_options.begin(), after.common_generation_options.end()) ==
        advertised_common
    );
    ASSERT_TRUE(before.provider_generation_options.empty());
    ASSERT_TRUE(after.provider_generation_options.empty());
    ASSERT_TRUE(!engine.loaded_model_info().has_value()); // model-free by design
    engine.shutdown();
}

// Content controls are inert on a test double whose output makes no content
// claims (user decision 2026-07-17 amending the no-silent-discard posture for
// Echo): accept them so real request pipelines run unmodified, warn once per
// engine lifetime so the discard is not silent. Options addressed to Echo by
// namespace stay demands: unknown keys are typos and reject.
TEST(EchoEngine, Echo_ignores_content_controls_with_one_warning) {
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    CollectingLog logs;
    ASSERT_TRUE(!engine.initialize(config, logs.logger(), {}).has_value());
    const std::string ignored_warning = "Ignoring content controls; echoed output makes no content claims";

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "hi";
    req.gen_config.temperature = 0.5f;
    req.gen_config.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= \"x\""};
    req.chat_template = "{{ ignored }}";
    req.gen_config.show_thinking = true;
    req.gen_config.provider_options["llama"] = Chorus::ProviderOptionMap{{"repeat_penalty", 1.1}};
    ASSERT_TRUE(!engine.validate_request(req).has_value());

    // Second sighting stays quiet: one warning per engine lifetime.
    ASSERT_TRUE(!engine.validate_request(req).has_value());
    ASSERT_EQ(logs.count(ignored_warning), size_t{1});

    // The discarded controls ride a field, not the sentence: a host can list
    // them without parsing the message back apart.
    const auto records = logs.records();
    const auto* controls = find_log_field(records.front(), "controls");
    ASSERT_TRUE(controls != nullptr);
    const auto* control_names = controls ? std::get_if<std::string>(controls) : nullptr;
    ASSERT_TRUE(control_names != nullptr);
    if (control_names) {
        ASSERT_TRUE(control_names->find("temperature") != std::string::npos);
        ASSERT_TRUE(control_names->find("chat_template") != std::string::npos);
        ASSERT_TRUE(control_names->find("show_thinking") != std::string::npos);
    }

    // Namespace-addressed options are demands, not content controls.
    req = {};
    req.id = 3;
    req.prompt = "hi";
    req.gen_config.provider_options["echo"] = Chorus::ProviderOptionMap{{"volume", int64_t{11}}};
    auto rejection = engine.validate_request(req);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    engine.shutdown();
}

TEST(EchoEngine, Echo_accepts_session_id_as_correlation) {
    // Correlation is universal: native_sessions=false must not reject it.
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());
    Chorus::ChorusRequest req;
    req.id = 4;
    req.prompt = "hi";
    req.session_id = "npc_42/dialogue";
    ASSERT_TRUE(!engine.validate_request(req).has_value());
    ASSERT_TRUE(!engine.capabilities().native_sessions);
    engine.shutdown();
}

TEST(EchoEngine, Echo_max_tokens_counts_word_chunks) {
    const auto run = [](int64_t id, int32_t max_tokens) {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> signals;

        Chorus::EchoEngine engine;
        Chorus::ChorusConfig config;
        if (engine.initialize(config, {}, {}).has_value()) {
            ADD_FAILURE() << "Echo engine initialization failed.";
            return std::vector<Chorus::ChorusSignal>{};
        }

        Chorus::ChorusRequest request;
        request.id = id;
        request.prompt = "one two three";
        request.gen_config.max_tokens = max_tokens;
        if (engine.validate_request(request).has_value()) {
            ADD_FAILURE() << "Echo rejected a valid max_tokens request.";
            engine.shutdown();
            return std::vector<Chorus::ChorusSignal>{};
        }
        request.on_event = [&](Chorus::ChorusSignal& signal) {
            std::lock_guard<std::mutex> lock(mutex);
            signals.push_back(signal);
            cv.notify_all();
        };
        engine.submit_request(request);

        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!cv.wait_for(lock, std::chrono::seconds(2), [&] {
                    return !signals.empty() && std::holds_alternative<Chorus::ChorusSignal::Stop>(signals.back().event);
                })) {
                ADD_FAILURE() << "Timed out waiting for Echo generation.";
                lock.unlock();
                engine.shutdown();
                return std::vector<Chorus::ChorusSignal>{};
            }
        }
        engine.shutdown();
        return signals;
    };

    const auto zero = run(10, 0);
    ASSERT_EQ(zero.size(), size_t{1});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(zero[0].event));

    const auto one = run(11, 1);
    ASSERT_EQ(one.size(), size_t{2});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(one[0].event));
    ASSERT_EQ(std::get<Chorus::ChorusSignal::Token>(one[0].event).text, std::string("one "));
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(one[1].event));

    const auto oversized = run(12, 8);
    ASSERT_EQ(oversized.size(), size_t{4});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(oversized[0].event));
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(oversized[1].event));
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(oversized[2].event));
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Stop>(oversized[3].event));

    const auto unbounded = run(13, -1);
    ASSERT_EQ(unbounded.size(), size_t{4});
}

TEST(EchoEngine, Echo_rejects_max_tokens_below_negative_sentinel) {
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest request;
    request.id = 14;
    request.prompt = "hi";
    request.gen_config.max_tokens = -2;
    const auto rejection = engine.validate_request(request);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    engine.shutdown();
}

TEST(EchoEngine, Echo_cancels_active_request_reentrantly_once) {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest request;
    request.id = 20;
    request.prompt = "first second third";
    request.on_event = [&](Chorus::ChorusSignal& signal) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            signals.push_back(signal);
        }
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            engine.cancel_request(request.id);
            engine.cancel_request(request.id);
            engine.cancel_request(9999);
        }
        cv.notify_all();
    };
    engine.submit_request(request);

    bool terminal_reached = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        terminal_reached = cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return !signals.empty() && std::holds_alternative<Chorus::ChorusSignal::Error>(signals.back().event);
        });
    }
    engine.shutdown();
    ASSERT_TRUE(terminal_reached);

    ASSERT_EQ(signals.size(), size_t{2});
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Token>(signals[0].event));
    ASSERT_TRUE(std::holds_alternative<Chorus::ChorusSignal::Error>(signals[1].event));
    ASSERT_TRUE(std::get<Chorus::ChorusSignal::Error>(signals[1].event).code == Chorus::ChorusError::Cancelled);
    ASSERT_EQ(signals[1].request_id, request.id);
    const size_t terminals = std::count_if(signals.begin(), signals.end(), [](const auto& signal) {
        return std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event) ||
               std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
    });
    ASSERT_EQ(terminals, size_t{1});
}

TEST(EchoEngine, Echo_cancels_queued_request_without_affecting_another) {
    std::mutex mutex;
    std::condition_variable cv;
    bool active_callback_blocked = false;
    bool release_active = false;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest active;
    active.id = 30;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        std::unique_lock<std::mutex> lock(mutex);
        signals.push_back(signal);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event) && !active_callback_blocked) {
            active_callback_blocked = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_active; });
        }
        cv.notify_all();
    };
    engine.submit_request(active);

    bool callback_reached = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        callback_reached = cv.wait_for(lock, std::chrono::seconds(2), [&] { return active_callback_blocked; });
        if (!callback_reached) {
            release_active = true;
            cv.notify_all();
        }
    }
    if (!callback_reached) {
        engine.shutdown();
        ASSERT_TRUE(callback_reached);
    }

    Chorus::ChorusRequest cancelled;
    cancelled.id = 31;
    cancelled.prompt = "queued victim";
    cancelled.on_event = [&](Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        signals.push_back(signal);
        cv.notify_all();
    };
    Chorus::ChorusRequest survivor = cancelled;
    survivor.id = 32;
    survivor.prompt = "queued survivor";
    engine.submit_request(cancelled);
    engine.submit_request(survivor);
    engine.cancel_request(cancelled.id);
    engine.cancel_request(cancelled.id);
    engine.cancel_request(9999);

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_active = true;
        cv.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] {
            for (const auto& signal : signals)
                if (signal.request_id == survivor.id && std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event))
                    return true;
            return false;
        }));
    }
    engine.shutdown();

    size_t cancelled_terminals = 0;
    size_t cancelled_nonterminals = 0;
    bool survivor_stopped = false;
    for (const auto& signal : signals) {
        if (signal.request_id == cancelled.id) {
            cancelled_terminals += std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
            cancelled_nonterminals += !std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event);
            ASSERT_TRUE(std::get<Chorus::ChorusSignal::Error>(signal.event).code == Chorus::ChorusError::Cancelled);
        }
        if (signal.request_id == survivor.id && std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event))
            survivor_stopped = true;
    }
    ASSERT_EQ(cancelled_terminals, size_t{1});
    ASSERT_EQ(cancelled_nonterminals, size_t{0});
    ASSERT_TRUE(survivor_stopped);
}

TEST(EchoEngine, Echo_shutdown_drains_active_and_queued_requests_before_returning) {
    std::mutex mutex;
    std::condition_variable cv;
    bool active_callback_blocked = false;
    bool release_active = false;
    bool shutdown_started = false;
    bool shutdown_returned = false;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest active;
    active.id = 40;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        std::unique_lock<std::mutex> lock(mutex);
        signals.push_back(signal);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event) && !active_callback_blocked) {
            active_callback_blocked = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_active; });
        }
    };
    engine.submit_request(active);

    bool callback_reached = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        callback_reached = cv.wait_for(lock, std::chrono::seconds(2), [&] { return active_callback_blocked; });
        if (!callback_reached) {
            release_active = true;
            cv.notify_all();
        }
    }
    if (!callback_reached) {
        engine.shutdown();
        ASSERT_TRUE(callback_reached);
    }

    Chorus::ChorusRequest queued = active;
    queued.id = 41;
    queued.prompt = "queued request";
    queued.on_event = [&](Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        signals.push_back(signal);
    };
    engine.submit_request(queued);

    std::thread stopper([&] {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown_started = true;
            cv.notify_all();
        }
        engine.shutdown();
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown_returned = true;
            cv.notify_all();
        }
    });
    bool stopper_started = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        stopper_started = cv.wait_for(lock, std::chrono::seconds(2), [&] { return shutdown_started; });
        release_active = true;
        cv.notify_all();
    }
    stopper.join();

    size_t active_cancelled = 0;
    size_t queued_cancelled = 0;
    size_t active_tokens = 0;
    size_t queued_tokens = 0;
    size_t stops = 0;
    for (const auto& signal : signals) {
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event)) {
            active_tokens += signal.request_id == active.id;
            queued_tokens += signal.request_id == queued.id;
            continue;
        }
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(signal.event)) {
            ++stops;
            continue;
        }
        if (std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event)) {
            ASSERT_TRUE(std::get<Chorus::ChorusSignal::Error>(signal.event).code == Chorus::ChorusError::Cancelled);
            active_cancelled += signal.request_id == active.id;
            queued_cancelled += signal.request_id == queued.id;
        }
    }
    ASSERT_EQ(active_tokens, size_t{1});
    ASSERT_EQ(queued_tokens, size_t{0});
    ASSERT_EQ(stops, size_t{0});
    ASSERT_EQ(active_cancelled, size_t{1});
    ASSERT_EQ(queued_cancelled, size_t{1});
    ASSERT_TRUE(stopper_started);
    ASSERT_TRUE(shutdown_returned);
}

TEST(EchoEngine, Echo_shutdown_waits_for_queued_cancellation_callback) {
    std::mutex mutex;
    std::condition_variable cv;
    bool active_callback_blocked = false;
    bool release_active = false;
    bool cancellation_callback_blocked = false;
    bool release_cancellation = false;
    bool stop_returned = false;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config, {}, {}).has_value());

    Chorus::ChorusRequest active;
    active.id = 50;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event))
            return;
        std::unique_lock<std::mutex> lock(mutex);
        if (active_callback_blocked)
            return;
        active_callback_blocked = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_active; });
    };
    engine.submit_request(active);

    bool active_reached = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        active_reached = cv.wait_for(lock, std::chrono::seconds(2), [&] { return active_callback_blocked; });
        if (!active_reached) {
            release_active = true;
            cv.notify_all();
        }
    }
    if (!active_reached) {
        engine.shutdown();
        ASSERT_TRUE(active_reached);
    }

    Chorus::ChorusRequest queued;
    queued.id = 51;
    queued.prompt = "queued request";
    queued.on_event = [&](Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event))
            return;
        std::unique_lock<std::mutex> lock(mutex);
        cancellation_callback_blocked = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_cancellation; });
    };
    engine.submit_request(queued);

    std::thread cancel([&] { engine.cancel_request(queued.id); });
    bool cancellation_reached = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        cancellation_reached =
            cv.wait_for(lock, std::chrono::seconds(2), [&] { return cancellation_callback_blocked; });
        if (!cancellation_reached) {
            release_active = true;
            release_cancellation = true;
            cv.notify_all();
        }
    }
    if (!cancellation_reached) {
        cancel.join();
        engine.shutdown();
        ASSERT_TRUE(cancellation_reached);
    }

    std::thread stopper([&] {
        engine.shutdown();
        std::lock_guard<std::mutex> lock(mutex);
        stop_returned = true;
        cv.notify_all();
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_active = true;
        cv.notify_all();
    }

    bool returned_while_callback_blocked = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        returned_while_callback_blocked =
            cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return stop_returned; });
        release_cancellation = true;
        cv.notify_all();
    }
    cancel.join();
    stopper.join();

    ASSERT_TRUE(!returned_while_callback_blocked);
    ASSERT_TRUE(stop_returned);
}

namespace {

constexpr std::string_view ECHO_REENTRANT_SHUTDOWN_CHILD = "echo_reentrant_shutdown";

int run_echo_reentrant_shutdown_child() {
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool active_callback_blocked = false;
        bool release_active = false;
        bool cancellation_callback_entered = false;
        bool shutdown_returned = false;
        size_t cancelled_terminals = 0;
    } state;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    if (engine.initialize(config, {}, {}).has_value())
        return 10;

    Chorus::ChorusRequest active;
    active.id = 52;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        if (!std::holds_alternative<Chorus::ChorusSignal::Token>(signal.event))
            return;
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.active_callback_blocked)
            return;
        state.active_callback_blocked = true;
        state.cv.notify_all();
        state.cv.wait(lock, [&] { return state.release_active; });
    };
    engine.submit_request(active);

    {
        std::unique_lock<std::mutex> lock(state.mutex);
        if (!state.cv.wait_for(lock, std::chrono::seconds(2), [&] { return state.active_callback_blocked; })) {
            state.release_active = true;
            state.cv.notify_all();
            lock.unlock();
            engine.shutdown();
            return 11;
        }
    }

    Chorus::ChorusRequest queued;
    queued.id = 53;
    queued.prompt = "queued request";
    queued.on_event = [&](Chorus::ChorusSignal& signal) {
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.cancelled_terminals +=
                std::holds_alternative<Chorus::ChorusSignal::Error>(signal.event) &&
                std::get<Chorus::ChorusSignal::Error>(signal.event).code == Chorus::ChorusError::Cancelled;
            state.cancellation_callback_entered = true;
            state.cv.notify_all();
        }
        engine.shutdown();
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.shutdown_returned = true;
        }
    };
    engine.submit_request(queued);

    std::thread release([&] {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.cv.wait(lock, [&] { return state.cancellation_callback_entered; });
        state.release_active = true;
        state.cv.notify_all();
    });
    engine.cancel_request(queued.id);
    release.join();

    bool passed = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        passed = state.shutdown_returned && state.cancelled_terminals == 1;
    }
    return passed && !engine.is_initialized() ? 0 : 12;
}

} // namespace

TEST(EchoEngine, Echo_queued_cancellation_callback_can_reenter_shutdown) {
    ASSERT_TRUE(
        run_isolated_test_child(std::string(ECHO_REENTRANT_SHUTDOWN_CHILD), std::chrono::seconds(10))
    );
}

int run_echo_engine_child_mode(std::string_view child_name) {
    if (child_name == ECHO_REENTRANT_SHUTDOWN_CHILD)
        return run_echo_reentrant_shutdown_child();
    return 64;
}

TEST(EchoEngine, Echo_messages_echoes_last_user_message) {
    Chorus::EchoEngine engine;
    ASSERT_TRUE(!engine.initialize(Chorus::ChorusConfig{}, {}, {}).has_value());

    std::mutex mutex;
    std::condition_variable cv;
    std::string text;
    bool stopped = false;
    size_t terminals = 0;
    Chorus::ChorusRequest request;
    request.id = 1;
    request.messages = {
        {Chorus::MessageRole::System, Chorus::MessageContent::text("persona")},
        {Chorus::MessageRole::User, Chorus::MessageContent::text("first")},
        {Chorus::MessageRole::Assistant, Chorus::MessageContent::text("reply")},
        {Chorus::MessageRole::User, Chorus::MessageContent::text("second question")},
    };
    request.on_event = [&](Chorus::ChorusSignal& sig) {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::holds_alternative<Chorus::ChorusSignal::Token>(sig.event))
            text += std::get<Chorus::ChorusSignal::Token>(sig.event).text;
        if (std::holds_alternative<Chorus::ChorusSignal::Stop>(sig.event)) {
            stopped = true;
            ++terminals;
            cv.notify_all();
        } else if (std::holds_alternative<Chorus::ChorusSignal::Error>(sig.event)) {
            ++terminals;
            cv.notify_all();
        }
    };
    engine.submit_request(request);
    bool completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        completed = cv.wait_for(lock, std::chrono::seconds(2), [&] { return terminals == 1; });
    }
    engine.shutdown();

    ASSERT_TRUE(completed);
    ASSERT_EQ(terminals, size_t{1});
    ASSERT_TRUE(stopped); // completed, not cancelled
    ASSERT_EQ(text, std::string("second question"));
}

// Echo's binding to the shared port contract. Model-free, so it runs in every
// configuration and is the fast canary for a contract change.
static EngineUnderTest echo_under_test() {
    EngineUnderTest subject;
    subject.label = "Echo";
    subject.make_engine = [] { return std::make_unique<Chorus::EchoEngine>(); };
    subject.make_config = [] { return Chorus::ChorusConfig{}; }; // needs no model
    subject.shape_long_request = [](Chorus::ChorusRequest& request) {
        request.prompt = "one two three four five"; // five word chunks to stream
    };
    subject.shape_short_request = [](Chorus::ChorusRequest& request) { request.prompt = "hi"; };
    return subject;
}

INSTANTIATE_TEST_SUITE_P(
    Echo,
    EngineContractTest,
    ::testing::Values(echo_under_test()),
    [](const ::testing::TestParamInfo<EngineUnderTest>& info) { return info.param.label; }
);
