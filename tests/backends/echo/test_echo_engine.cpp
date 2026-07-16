#include "chorus/backends/echo/echo_engine.hpp"
#include "chorus/core/common.hpp"
#include "test_utils.hpp"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// No model, no skips: this suite must pass under CHORUS_SKIP_MODEL_TESTS=1.

void test_echo_initializes_without_a_model_file() {
    Chorus::EchoEngine engine;
    ASSERT_TRUE(!engine.is_initialized());

    Chorus::ChorusConfig config; // empty ModelSpec deliberately: Echo needs no model
    ASSERT_TRUE(!engine.initialize(config).has_value());
    ASSERT_TRUE(engine.is_initialized());

    engine.stop();
    ASSERT_TRUE(!engine.is_initialized());
}

void test_echo_streams_prompt_word_by_word_then_stops() {
    std::mutex sig_mutex;
    std::vector<Chorus::ChorusSignal> sigs;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

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
            if (!sigs.empty() && sigs.back().type == Chorus::EventType::Stop)
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
        ASSERT_TRUE(sigs[i].type == Chorus::EventType::Token);
        ASSERT_EQ(sigs[i].request_id, 7);
        reassembled += sigs[i].text;
    }
    ASSERT_TRUE(reassembled == "hello chorus seam");
    ASSERT_TRUE(sigs.back().type == Chorus::EventType::Stop);
    ASSERT_EQ(sigs.back().request_id, 7);

    engine.stop();
}

void test_echo_submit_before_initialize_signals_engine_not_ready() {
    // Pinned in the spec (2026-07-08): pre-init submit mirrors LlamaEngine exactly,
    // so a test passing on Echo cannot silently fail on Llama.
    Chorus::EchoEngine engine;

    bool errored = false;
    Chorus::ChorusError code = Chorus::ChorusError::None;

    Chorus::ChorusRequest req;
    req.id = 3;
    req.prompt = "ignored";
    req.on_event = [&](Chorus::ChorusSignal& sig) {
        // pre-init failure is emitted synchronously on the caller thread
        errored = sig.is_error();
        code = sig.error_code;
    };
    engine.submit_request(req);

    ASSERT_TRUE(errored);
    ASSERT_TRUE(code == Chorus::ChorusError::EngineNotReady);
}

void test_echo_capabilities_deterministic_across_init() {
    Chorus::EchoEngine engine;
    auto before = engine.capabilities();
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());
    auto after = engine.capabilities();
    ASSERT_EQ(before.backend_id, std::string("echo"));
    ASSERT_EQ(after.backend_id, before.backend_id);
    ASSERT_TRUE(before.streaming && after.streaming);
    ASSERT_TRUE(before.scheduling == Chorus::SchedulingAuthority::BackendManaged);
    ASSERT_TRUE(before.cancellation && after.cancellation);
    ASSERT_EQ(before.portable_generation_options.size(), size_t{1});
    ASSERT_EQ(before.portable_generation_options[0], std::string("max_tokens"));
    ASSERT_TRUE(after.portable_generation_options == before.portable_generation_options);
    ASSERT_TRUE(!engine.loaded_model_info().has_value()); // model-free by design
    engine.stop();
}

void test_echo_rejects_set_but_unsupported_controls() {
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest req;
    req.id = 1;
    req.prompt = "hi";

    req.gen_config.common.temperature = 0.5f; // set but unsupported
    auto r1 = engine.validate_request(req);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r1->error == Chorus::ChorusError::UnsupportedOption);

    req = {};
    req.id = 2;
    req.prompt = "hi";
    req.gen_config.common.constraint = Chorus::OutputConstraint{Chorus::ConstraintFormat::Gbnf, "root ::= \"x\""};
    auto r2 = engine.validate_request(req);
    ASSERT_TRUE(r2.has_value());
    ASSERT_TRUE(r2->error == Chorus::ChorusError::UnsupportedFeature);

    req = {};
    req.id = 3;
    req.prompt = "hi";
    req.gen_config.backend_options["echo"] = Chorus::OptionMap{{"volume", int64_t{11}}};
    auto r3 = engine.validate_request(req);
    ASSERT_TRUE(r3.has_value());
    ASSERT_TRUE(r3->error == Chorus::ChorusError::UnsupportedOption);
    engine.stop();
}

void test_echo_accepts_session_id_as_correlation() {
    // Correlation is universal: native_sessions=false must not reject it.
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());
    Chorus::ChorusRequest req;
    req.id = 4;
    req.prompt = "hi";
    req.session_id = "npc_42/dialogue";
    ASSERT_TRUE(!engine.validate_request(req).has_value());
    ASSERT_TRUE(!engine.capabilities().native_sessions);
    engine.stop();
}

void test_echo_max_tokens_counts_word_chunks() {
    const auto run = [](int64_t id, int32_t max_tokens) {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> signals;

        Chorus::EchoEngine engine;
        Chorus::ChorusConfig config;
        if (engine.initialize(config).has_value()) {
            g_tests_failed++;
            return std::vector<Chorus::ChorusSignal>{};
        }

        Chorus::ChorusRequest request;
        request.id = id;
        request.prompt = "one two three";
        request.gen_config.common.max_tokens = max_tokens;
        if (engine.validate_request(request).has_value()) {
            g_tests_failed++;
            engine.stop();
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
                    return !signals.empty() && signals.back().type == Chorus::EventType::Stop;
                })) {
                g_tests_failed++;
                lock.unlock();
                engine.stop();
                return std::vector<Chorus::ChorusSignal>{};
            }
        }
        engine.stop();
        return signals;
    };

    const auto zero = run(10, 0);
    ASSERT_EQ(zero.size(), size_t{1});
    ASSERT_TRUE(zero[0].type == Chorus::EventType::Stop);

    const auto one = run(11, 1);
    ASSERT_EQ(one.size(), size_t{2});
    ASSERT_TRUE(one[0].type == Chorus::EventType::Token);
    ASSERT_EQ(one[0].text, std::string("one "));
    ASSERT_TRUE(one[1].type == Chorus::EventType::Stop);

    const auto oversized = run(12, 8);
    ASSERT_EQ(oversized.size(), size_t{4});
    ASSERT_TRUE(oversized[0].type == Chorus::EventType::Token);
    ASSERT_TRUE(oversized[1].type == Chorus::EventType::Token);
    ASSERT_TRUE(oversized[2].type == Chorus::EventType::Token);
    ASSERT_TRUE(oversized[3].type == Chorus::EventType::Stop);

    const auto unbounded = run(13, -1);
    ASSERT_EQ(unbounded.size(), size_t{4});
}

void test_echo_rejects_max_tokens_below_negative_sentinel() {
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest request;
    request.id = 14;
    request.prompt = "hi";
    request.gen_config.common.max_tokens = -2;
    const auto rejection = engine.validate_request(request);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::UnsupportedOption);
    engine.stop();
}

void test_echo_cancels_active_request_reentrantly_once() {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest request;
    request.id = 20;
    request.prompt = "first second third";
    request.on_event = [&](Chorus::ChorusSignal& signal) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            signals.push_back(signal);
        }
        if (signal.type == Chorus::EventType::Token) {
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
            return !signals.empty() && signals.back().type == Chorus::EventType::Error;
        });
    }
    engine.stop();
    ASSERT_TRUE(terminal_reached);

    ASSERT_EQ(signals.size(), size_t{2});
    ASSERT_TRUE(signals[0].type == Chorus::EventType::Token);
    ASSERT_TRUE(signals[1].type == Chorus::EventType::Error);
    ASSERT_TRUE(signals[1].error_code == Chorus::ChorusError::Cancelled);
    ASSERT_EQ(signals[1].request_id, request.id);
}

void test_echo_cancels_queued_request_without_affecting_another() {
    std::mutex mutex;
    std::condition_variable cv;
    bool active_callback_blocked = false;
    bool release_active = false;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest active;
    active.id = 30;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        std::unique_lock<std::mutex> lock(mutex);
        signals.push_back(signal);
        if (signal.type == Chorus::EventType::Token && !active_callback_blocked) {
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
        engine.stop();
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
                if (signal.request_id == survivor.id && signal.type == Chorus::EventType::Stop)
                    return true;
            return false;
        }));
    }
    engine.stop();

    size_t cancelled_terminals = 0;
    size_t cancelled_nonterminals = 0;
    bool survivor_stopped = false;
    for (const auto& signal : signals) {
        if (signal.request_id == cancelled.id) {
            cancelled_terminals += signal.type == Chorus::EventType::Error;
            cancelled_nonterminals += signal.type != Chorus::EventType::Error;
            ASSERT_TRUE(signal.error_code == Chorus::ChorusError::Cancelled);
        }
        if (signal.request_id == survivor.id && signal.type == Chorus::EventType::Stop)
            survivor_stopped = true;
    }
    ASSERT_EQ(cancelled_terminals, size_t{1});
    ASSERT_EQ(cancelled_nonterminals, size_t{0});
    ASSERT_TRUE(survivor_stopped);
}

void test_echo_stop_drains_active_and_queued_requests_before_returning() {
    std::mutex mutex;
    std::condition_variable cv;
    bool active_callback_blocked = false;
    bool release_active = false;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest active;
    active.id = 40;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        std::unique_lock<std::mutex> lock(mutex);
        signals.push_back(signal);
        if (signal.type == Chorus::EventType::Token && !active_callback_blocked) {
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
        engine.stop();
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

    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::lock_guard<std::mutex> lock(mutex);
        release_active = true;
        cv.notify_all();
    });
    engine.stop();
    release.join();

    size_t active_cancelled = 0;
    size_t queued_cancelled = 0;
    size_t active_tokens = 0;
    size_t queued_tokens = 0;
    size_t stops = 0;
    for (const auto& signal : signals) {
        if (signal.type == Chorus::EventType::Token) {
            active_tokens += signal.request_id == active.id;
            queued_tokens += signal.request_id == queued.id;
            continue;
        }
        if (signal.type == Chorus::EventType::Stop) {
            ++stops;
            continue;
        }
        if (signal.type == Chorus::EventType::Error) {
            ASSERT_TRUE(signal.error_code == Chorus::ChorusError::Cancelled);
            active_cancelled += signal.request_id == active.id;
            queued_cancelled += signal.request_id == queued.id;
        }
    }
    ASSERT_EQ(active_tokens, size_t{1});
    ASSERT_EQ(queued_tokens, size_t{0});
    ASSERT_EQ(stops, size_t{0});
    ASSERT_EQ(active_cancelled, size_t{1});
    ASSERT_EQ(queued_cancelled, size_t{1});
}

void test_echo_stop_waits_for_queued_cancellation_callback() {
    std::mutex mutex;
    std::condition_variable cv;
    bool active_callback_blocked = false;
    bool release_active = false;
    bool cancellation_callback_blocked = false;
    bool release_cancellation = false;
    bool stop_returned = false;

    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    Chorus::ChorusRequest active;
    active.id = 50;
    active.prompt = "active request";
    active.on_event = [&](Chorus::ChorusSignal& signal) {
        if (signal.type != Chorus::EventType::Token)
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
        engine.stop();
        ASSERT_TRUE(active_reached);
    }

    Chorus::ChorusRequest queued;
    queued.id = 51;
    queued.prompt = "queued request";
    queued.on_event = [&](Chorus::ChorusSignal& signal) {
        if (signal.type != Chorus::EventType::Error)
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
        engine.stop();
        ASSERT_TRUE(cancellation_reached);
    }

    std::thread stopper([&] {
        engine.stop();
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

// Capability conformance: Echo advertises exactly one option, max_tokens, and no
// backend options. The matrix proves that option's deterministic behavior and pins
// both completeness directions against the engine's own advertisement.
void test_echo_conformance_matrix() {
    Chorus::EchoEngine engine;
    Chorus::ChorusConfig config;
    ASSERT_TRUE(!engine.initialize(config).has_value());

    const auto caps = engine.capabilities();
    ASSERT_EQ(caps.portable_generation_options.size(), size_t{1});
    ASSERT_EQ(caps.portable_generation_options[0], std::string("max_tokens"));
    ASSERT_TRUE(caps.backend_generation_options.empty());

    // Prove max_tokens deterministic behavior: capping the emitted word chunks.
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Chorus::ChorusSignal> signals;

    Chorus::ChorusRequest request;
    request.id = 60;
    request.prompt = "one two three";
    request.gen_config.common.max_tokens = 1;
    ASSERT_TRUE(!engine.validate_request(request).has_value());
    request.on_event = [&](Chorus::ChorusSignal& signal) {
        std::lock_guard<std::mutex> lock(mutex);
        signals.push_back(signal);
        cv.notify_all();
    };
    engine.submit_request(request);
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return !signals.empty() && signals.back().type == Chorus::EventType::Stop;
        }));
    }
    ASSERT_EQ(signals.size(), size_t{2});
    ASSERT_TRUE(signals[0].type == Chorus::EventType::Token);
    ASSERT_EQ(signals[0].text, std::string("one "));
    ASSERT_TRUE(signals[1].type == Chorus::EventType::Stop);

    std::set<std::string> covered{"max_tokens"};
    std::set<std::string> advertised(caps.portable_generation_options.begin(), caps.portable_generation_options.end());
    for (const auto& name : advertised)
        ASSERT_TRUE(covered.count(name) == 1); // every advertised option has a case
    for (const auto& name : covered)
        ASSERT_TRUE(advertised.count(name) == 1); // no case names an unadvertised option
    ASSERT_EQ(covered.size(), advertised.size());
    engine.stop();
}

// Terminal invariant at the Echo layer: every accepted request produces exactly one
// terminal event across success, reentrant cancellation, and engine stop.
void test_echo_terminal_invariant_one_per_request() {
    // Success: a normal completion ends with exactly one Stop.
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> signals;

        Chorus::EchoEngine engine;
        Chorus::ChorusConfig config;
        ASSERT_TRUE(!engine.initialize(config).has_value());

        Chorus::ChorusRequest request;
        request.id = 61;
        request.prompt = "alpha beta";
        request.on_event = [&](Chorus::ChorusSignal& signal) {
            std::lock_guard<std::mutex> lock(mutex);
            signals.push_back(signal);
            cv.notify_all();
        };
        engine.submit_request(request);
        {
            std::unique_lock<std::mutex> lock(mutex);
            ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] {
                return !signals.empty() && signals.back().type == Chorus::EventType::Stop;
            }));
        }
        engine.stop();
        size_t terminals = 0;
        for (const auto& signal : signals)
            terminals += signal.type == Chorus::EventType::Stop || signal.type == Chorus::EventType::Error;
        ASSERT_EQ(terminals, size_t{1});
        ASSERT_TRUE(signals.back().type == Chorus::EventType::Stop);
    }

    // Cancellation: cancelling mid-stream ends with exactly one Cancelled error.
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<Chorus::ChorusSignal> signals;

        Chorus::EchoEngine engine;
        Chorus::ChorusConfig config;
        ASSERT_TRUE(!engine.initialize(config).has_value());

        Chorus::ChorusRequest request;
        request.id = 62;
        request.prompt = "first second third";
        request.on_event = [&](Chorus::ChorusSignal& signal) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                signals.push_back(signal);
            }
            if (signal.type == Chorus::EventType::Token)
                engine.cancel_request(request.id);
            cv.notify_all();
        };
        engine.submit_request(request);
        {
            std::unique_lock<std::mutex> lock(mutex);
            ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] {
                return !signals.empty() && signals.back().type == Chorus::EventType::Error;
            }));
        }
        engine.stop();
        size_t terminals = 0;
        for (const auto& signal : signals)
            terminals += signal.type == Chorus::EventType::Stop || signal.type == Chorus::EventType::Error;
        ASSERT_EQ(terminals, size_t{1});
        ASSERT_TRUE(signals.back().type == Chorus::EventType::Error);
        ASSERT_TRUE(signals.back().error_code == Chorus::ChorusError::Cancelled);
    }

    // Engine stop: stopping while a request is queued drains it with exactly one
    // Cancelled terminal (the active callback is held so the request stays in flight).
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool active_blocked = false;
        bool release_active = false;
        std::vector<Chorus::ChorusSignal> signals;

        Chorus::EchoEngine engine;
        Chorus::ChorusConfig config;
        ASSERT_TRUE(!engine.initialize(config).has_value());

        Chorus::ChorusRequest active;
        active.id = 63;
        active.prompt = "active request";
        active.on_event = [&](Chorus::ChorusSignal& signal) {
            std::unique_lock<std::mutex> lock(mutex);
            signals.push_back(signal);
            if (signal.type == Chorus::EventType::Token && !active_blocked) {
                active_blocked = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_active; });
            }
        };
        engine.submit_request(active);

        bool reached = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            reached = cv.wait_for(lock, std::chrono::seconds(2), [&] { return active_blocked; });
            if (!reached) {
                release_active = true;
                cv.notify_all();
            }
        }
        if (!reached) {
            engine.stop();
            ASSERT_TRUE(reached);
        }

        Chorus::ChorusRequest queued;
        queued.id = 64;
        queued.prompt = "queued request";
        queued.on_event = [&](Chorus::ChorusSignal& signal) {
            std::lock_guard<std::mutex> lock(mutex);
            signals.push_back(signal);
        };
        engine.submit_request(queued);

        std::thread release([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::lock_guard<std::mutex> lock(mutex);
            release_active = true;
            cv.notify_all();
        });
        engine.stop();
        release.join();

        size_t active_terminals = 0;
        size_t queued_terminals = 0;
        for (const auto& signal : signals) {
            const bool terminal = signal.type == Chorus::EventType::Stop || signal.type == Chorus::EventType::Error;
            if (!terminal)
                continue;
            ASSERT_TRUE(signal.type == Chorus::EventType::Error);
            ASSERT_TRUE(signal.error_code == Chorus::ChorusError::Cancelled);
            active_terminals += signal.request_id == active.id;
            queued_terminals += signal.request_id == queued.id;
        }
        ASSERT_EQ(active_terminals, size_t{1});
        ASSERT_EQ(queued_terminals, size_t{1});
    }
}

int run_echo_engine_tests() {
    std::cout << "\n--- ECHO ENGINE SUITE ---\n";

    run_test("Echo_initializes_without_a_model_file", test_echo_initializes_without_a_model_file);
    run_test("Echo_streams_prompt_word_by_word_then_stops", test_echo_streams_prompt_word_by_word_then_stops);
    run_test(
        "Echo_submit_before_initialize_signals_EngineNotReady",
        test_echo_submit_before_initialize_signals_engine_not_ready
    );
    run_test("Echo_capabilities_deterministic_across_init", test_echo_capabilities_deterministic_across_init);
    run_test("Echo_rejects_set_but_unsupported_controls", test_echo_rejects_set_but_unsupported_controls);
    run_test("Echo_accepts_session_id_as_correlation", test_echo_accepts_session_id_as_correlation);
    run_test("Echo_max_tokens_counts_word_chunks", test_echo_max_tokens_counts_word_chunks);
    run_test("Echo_rejects_max_tokens_below_negative_sentinel", test_echo_rejects_max_tokens_below_negative_sentinel);
    run_test("Echo_cancels_active_request_reentrantly_once", test_echo_cancels_active_request_reentrantly_once);
    run_test(
        "Echo_cancels_queued_request_without_affecting_another",
        test_echo_cancels_queued_request_without_affecting_another
    );
    run_test(
        "Echo_stop_drains_active_and_queued_requests_before_returning",
        test_echo_stop_drains_active_and_queued_requests_before_returning
    );
    run_test("Echo_stop_waits_for_queued_cancellation_callback", test_echo_stop_waits_for_queued_cancellation_callback);
    run_test("Echo_conformance_matrix_covers_advertised_options", test_echo_conformance_matrix);
    run_test("Echo_terminal_invariant_one_per_request", test_echo_terminal_invariant_one_per_request);

    std::cout << "\n======================================\n";
    if (g_tests_failed > 0) {
        std::cout << RED << "SUMMARY: " << g_tests_failed << " FAILED, " << g_tests_passed << " PASSED." << RESET
                  << "\n";
        return 1;
    }
    std::cout << GREEN << "SUMMARY: ALL TESTS PASSED." << RESET << "\n";
    return 0;
}
