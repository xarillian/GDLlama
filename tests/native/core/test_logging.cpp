#include "chorus/core/log.hpp"
#include "collecting_sink.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// The diagnostics channel is deterministic and needs no model, which is the
// point of putting it in core: every claim below is checkable without a
// vendor library, a provider, or a host.

// ---------------------------------------------------------------------------
// The severity threshold
// ---------------------------------------------------------------------------

void test_a_level_below_the_threshold_produces_no_record() {
    CollectingSink sink;
    Chorus::Logger log = sink.logger("test", Chorus::log_level_default); // Warn and above

    log.debug("Swallowed");
    log.info("Also swallowed");
    log.warn("Kept");

    ASSERT_EQ(sink.size(), size_t{1});
    ASSERT_EQ(sink.records()[0].message, "Kept");
}

void test_enabled_reports_the_threshold_it_was_built_with() {
    CollectingSink sink;
    Chorus::Logger log = sink.logger("test", Chorus::log_level_default);

    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Debug));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Info));
    ASSERT_TRUE(log.enabled(Chorus::LogLevel::Warn));
    ASSERT_TRUE(log.enabled(Chorus::LogLevel::Error));
    ASSERT_TRUE(log.enabled(Chorus::LogLevel::Fatal));
}

// A provider that was never given a logger still runs; it just says nothing.
void test_a_default_constructed_logger_discards_everything() {
    Chorus::Logger log;
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Fatal));
    log.fatal("Nobody hears this");
    ASSERT_TRUE(true); // no sink, no crash
}

// Off is the one threshold that admits nothing, which is how a host asks for
// silence without the runtime having to special-case an absent logger.
void test_the_off_threshold_admits_nothing() {
    CollectingSink sink;
    Chorus::Logger log = sink.logger("test", Chorus::LogLevel::Off);

    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Debug));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Fatal));

    log.fatal("Unheard");
    ASSERT_EQ(sink.size(), size_t{0});
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

void test_a_record_carries_its_fields_typed() {
    CollectingSink sink;
    Chorus::Logger log = sink.logger("llama");

    log.error(
        "Decode failed", {{"code", -1}, {"sequence", 3}, {"fatal", true}, {"seconds", 0.5f}, {"stage", "decode"}}
    );

    const auto records = sink.records();
    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_EQ(records[0].source, "llama");
    ASSERT_TRUE(records[0].level == Chorus::LogLevel::Error);
    ASSERT_EQ(std::get<int64_t>(*find_log_field(records[0], "code")), int64_t{-1});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(records[0], "sequence")), int64_t{3});
    ASSERT_TRUE(std::get<bool>(*find_log_field(records[0], "fatal")));
    ASSERT_TRUE(std::get<double>(*find_log_field(records[0], "seconds")) == 0.5);
    ASSERT_EQ(std::get<std::string>(*find_log_field(records[0], "stage")), "decode");
}

// Two occurrences of one failure produce the same message and differ only in
// their fields, which is what lets a host group, filter, and count them.
void test_the_same_failure_twice_produces_the_same_message() {
    CollectingSink sink;
    Chorus::Logger log = sink.logger("llama");

    log.error("Decode failed", {{"code", -1}});
    log.error("Decode failed", {{"code", -2}});

    ASSERT_EQ(sink.count("Decode failed"), size_t{2});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(sink.records()[0], "code")), int64_t{-1});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(sink.records()[1], "code")), int64_t{-2});
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------

void test_a_for_request_logger_stamps_every_record_it_produces() {
    CollectingSink sink;
    Chorus::Logger base = sink.logger("llama");
    Chorus::Logger scoped = base.for_request(42, std::string("npc_7/dialogue"));

    scoped.warn("First");
    scoped.warn("Second");

    for (const auto& record : sink.records()) {
        ASSERT_TRUE(record.request_id.has_value());
        ASSERT_EQ(*record.request_id, int64_t{42});
        ASSERT_TRUE(record.session_id.has_value());
        ASSERT_EQ(*record.session_id, "npc_7/dialogue");
    }
}

void test_the_base_logger_is_unchanged_by_for_request() {
    CollectingSink sink;
    Chorus::Logger base = sink.logger("llama");
    (void)base.for_request(42, std::string("npc_7"));

    base.warn("Engine-wide");

    ASSERT_TRUE(!sink.records()[0].request_id.has_value());
    ASSERT_TRUE(!sink.records()[0].session_id.has_value());
}

void test_a_request_scoped_logger_may_name_no_session() {
    CollectingSink sink;
    Chorus::Logger scoped = sink.logger("llama").for_request(7);

    scoped.warn("Stateless");

    ASSERT_EQ(*sink.records()[0].request_id, int64_t{7});
    ASSERT_TRUE(!sink.records()[0].session_id.has_value());
}

// ---------------------------------------------------------------------------
// The synchronous stderr path
// ---------------------------------------------------------------------------

// Async delivery otherwise loses the last message before a crash, which is
// exactly the message worth having.
void test_a_severe_record_also_goes_to_stderr() {
    CollectingSink sink;
    std::stringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    sink.logger("llama").error("Decode failed", {{"code", -1}});
    sink.logger("llama").info("Routine");
    std::cerr.rdbuf(previous);

    ASSERT_TRUE(captured.str().find("Decode failed") != std::string::npos);
    ASSERT_TRUE(captured.str().find("Routine") == std::string::npos); // Info is not evidence
    ASSERT_EQ(sink.size(), size_t{2});                                // and the channel still sees both
}

// A producer fanning one record out to several loggers owes stderr one line,
// not one per subscriber, so it silences the copies and echoes once itself.
void test_without_stderr_echo_silences_the_synchronous_path_only() {
    CollectingSink sink;
    std::stringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    sink.logger("llama").without_stderr_echo().error("Decode failed");
    std::cerr.rdbuf(previous);

    ASSERT_EQ(captured.str(), std::string());
    ASSERT_EQ(sink.size(), size_t{1}); // the record itself is untouched
    ASSERT_EQ(sink.records()[0].message, "Decode failed");
}

// ---------------------------------------------------------------------------
// The channel
// ---------------------------------------------------------------------------

Chorus::LogRecord numbered_record(int64_t n) {
    Chorus::LogRecord record;
    record.message = "Record";
    record.fields = {{"n", n}};
    return record;
}

void test_the_channel_drains_in_production_order_and_empties() {
    Chorus::LogChannel channel(8);
    channel.push(numbered_record(1));
    channel.push(numbered_record(2));

    auto drained = channel.drain();
    ASSERT_EQ(drained.size(), size_t{2});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[0], "n")), int64_t{1});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[1], "n")), int64_t{2});
    ASSERT_EQ(channel.drain().size(), size_t{0});
}

// A full channel drops its oldest: the failure mode is a host that stopped
// polling, and after an hour of that you want the last second, not the first.
void test_overflow_drops_the_oldest_and_reports_the_count() {
    Chorus::LogChannel channel(2);
    channel.push(numbered_record(1));
    channel.push(numbered_record(2));
    channel.push(numbered_record(3));

    auto drained = channel.drain();
    ASSERT_EQ(drained.size(), size_t{3}); // one drop report plus two survivors
    ASSERT_EQ(drained[0].message, "Log records dropped");
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[0], "count")), int64_t{1});
    ASSERT_TRUE(drained[0].level == Chorus::LogLevel::Warn);
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[1], "n")), int64_t{2});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[2], "n")), int64_t{3});
}

// The count is per drain, not cumulative: a host is told what it missed since
// it last looked, not since the process started.
void test_the_drop_count_resets_after_it_surfaces() {
    Chorus::LogChannel channel(1);
    channel.push(numbered_record(1));
    channel.push(numbered_record(2));
    ASSERT_EQ(channel.drain().size(), size_t{2});

    channel.push(numbered_record(3));
    auto drained = channel.drain();
    ASSERT_EQ(drained.size(), size_t{1});
    ASSERT_EQ(drained[0].message, "Record");
}

void test_many_producer_threads_lose_nothing_within_capacity() {
    auto channel = std::make_shared<Chorus::LogChannel>(4096);
    Chorus::Logger log(Chorus::LogChannel::sink_for(channel), Chorus::LogLevel::Debug, "test");

    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&log] {
            for (int i = 0; i < 100; ++i)
                log.warn("Concurrent");
        });
    }
    for (auto& producer : producers)
        producer.join();

    auto drained = channel->drain();
    ASSERT_EQ(drained.size(), size_t{400});
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

void test_format_puts_identity_and_fields_in_one_line() {
    Chorus::LogRecord record;
    record.level = Chorus::LogLevel::Error;
    record.message = "Decode failed";
    record.fields = {{"code", -1}, {"sequence", 3}};
    record.request_id = 42;
    record.session_id = "npc_7";

    ASSERT_EQ(
        Chorus::format_log_record(record),
        "[Chorus] ERROR: Decode failed (request=42, session=npc_7, code=-1, sequence=3)"
    );
}

void test_format_omits_the_parenthesis_when_there_is_nothing_to_put_in_it() {
    Chorus::LogRecord record;
    record.level = Chorus::LogLevel::Info;
    record.message = "Engine is already initialized";

    ASSERT_EQ(Chorus::format_log_record(record), "[Chorus] INFO: Engine is already initialized");
}

int run_logging_tests() {
    std::cout << "\n--- Structured logging ---\n";

    run_test("Log_level_below_the_threshold_produces_no_record", test_a_level_below_the_threshold_produces_no_record);
    run_test(
        "Log_enabled_reports_the_threshold_it_was_built_with", test_enabled_reports_the_threshold_it_was_built_with
    );
    run_test(
        "Log_default_constructed_logger_discards_everything", test_a_default_constructed_logger_discards_everything
    );
    run_test("Log_off_threshold_admits_nothing", test_the_off_threshold_admits_nothing);

    run_test("Log_record_carries_its_fields_typed", test_a_record_carries_its_fields_typed);
    run_test("Log_same_failure_twice_produces_the_same_message", test_the_same_failure_twice_produces_the_same_message);

    run_test("Log_for_request_logger_stamps_every_record", test_a_for_request_logger_stamps_every_record_it_produces);
    run_test("Log_base_logger_is_unchanged_by_for_request", test_the_base_logger_is_unchanged_by_for_request);
    run_test("Log_request_scoped_logger_may_name_no_session", test_a_request_scoped_logger_may_name_no_session);

    run_test("Log_severe_record_also_goes_to_stderr", test_a_severe_record_also_goes_to_stderr);
    run_test(
        "Log_without_stderr_echo_silences_stderr_only", test_without_stderr_echo_silences_the_synchronous_path_only
    );

    run_test("Log_channel_drains_in_order_and_empties", test_the_channel_drains_in_production_order_and_empties);
    run_test("Log_channel_overflow_drops_oldest_and_reports", test_overflow_drops_the_oldest_and_reports_the_count);
    run_test("Log_channel_drop_count_resets_after_it_surfaces", test_the_drop_count_resets_after_it_surfaces);
    run_test("Log_channel_survives_many_producer_threads", test_many_producer_threads_lose_nothing_within_capacity);

    run_test("Log_format_puts_identity_and_fields_in_one_line", test_format_puts_identity_and_fields_in_one_line);
    run_test(
        "Log_format_omits_empty_parenthesis", test_format_omits_the_parenthesis_when_there_is_nothing_to_put_in_it
    );

    return g_tests_failed;
}
