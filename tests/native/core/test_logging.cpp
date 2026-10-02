#include "chorus/core/log.hpp"
#include "collecting_log.hpp"
#include "gtest_utils.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// The diagnostics channel is deterministic and needs no model, which is the
// point of putting it in core: every claim below is checkable without a
// vendor library, a provider, or a host.

// ---------------------------------------------------------------------------
// The severity threshold
// ---------------------------------------------------------------------------

TEST(Logging, Log_level_below_the_threshold_produces_no_record) {
    CollectingLog logs;
    Chorus::Logger log = logs.logger(Chorus::log_level_default); // Warn and above

    log.debug("Swallowed");
    log.info("Also swallowed");
    log.warn("Kept");

    ASSERT_EQ(logs.size(), size_t{1});
    ASSERT_EQ(logs.records()[0].message, "Kept");
}

TEST(Logging, Log_enabled_reports_the_threshold_it_was_built_with) {
    CollectingLog logs;
    Chorus::Logger log = logs.logger(Chorus::log_level_default);

    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Debug));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Info));
    ASSERT_TRUE(log.enabled(Chorus::LogLevel::Warn));
    ASSERT_TRUE(log.enabled(Chorus::LogLevel::Error));
    ASSERT_TRUE(log.enabled(Chorus::LogLevel::Fatal));
}

// A provider that was never given a logger still runs; it just says nothing.
TEST(Logging, Log_default_constructed_logger_discards_everything) {
    Chorus::Logger log;
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Debug));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Info));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Warn));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Error));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Fatal));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Off));

    log.fatal("Nobody hears this");
}

// Off is the one threshold that admits nothing, which is how a host asks for
// silence without the runtime having to special-case an absent logger.
TEST(Logging, Log_off_threshold_admits_nothing) {
    CollectingLog logs;
    Chorus::Logger log = logs.logger(Chorus::LogLevel::Off);

    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Debug));
    ASSERT_TRUE(!log.enabled(Chorus::LogLevel::Fatal));

    log.fatal("Unheard");
    ASSERT_EQ(logs.size(), size_t{0});
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

TEST(Logging, Log_record_carries_its_fields_typed) {
    CollectingLog logs;
    Chorus::Logger log = logs.logger();

    log.error(
        "Decode failed", {{"code", -1}, {"sequence", 3}, {"fatal", true}, {"seconds", 0.5f}, {"stage", "decode"}}
    );

    const auto records = logs.records();
    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_TRUE(records[0].level == Chorus::LogLevel::Error);
    ASSERT_EQ(std::get<int64_t>(*find_log_field(records[0], "code")), int64_t{-1});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(records[0], "sequence")), int64_t{3});
    ASSERT_TRUE(std::get<bool>(*find_log_field(records[0], "fatal")));
    ASSERT_TRUE(std::get<double>(*find_log_field(records[0], "seconds")) == 0.5);
    ASSERT_EQ(std::get<std::string>(*find_log_field(records[0], "stage")), "decode");
}

// Two occurrences of one failure produce the same message and differ only in
// their fields, which is what lets a host group, filter, and count them.
TEST(Logging, Log_same_failure_twice_produces_the_same_message) {
    CollectingLog logs;
    Chorus::Logger log = logs.logger();

    log.error("Decode failed", {{"code", -1}});
    log.error("Decode failed", {{"code", -2}});

    ASSERT_EQ(logs.count("Decode failed"), size_t{2});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(logs.records()[0], "code")), int64_t{-1});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(logs.records()[1], "code")), int64_t{-2});
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------

TEST(Logging, Log_for_request_logger_stamps_every_record) {
    CollectingLog logs;
    Chorus::Logger base = logs.logger();
    Chorus::Logger scoped = base.for_request(42, std::string("npc_7/dialogue"));

    scoped.warn("First");
    scoped.warn("Second");

    for (const auto& record : logs.records()) {
        ASSERT_TRUE(record.request_id.has_value());
        ASSERT_EQ(*record.request_id, int64_t{42});
        ASSERT_TRUE(record.session_id.has_value());
        ASSERT_EQ(*record.session_id, "npc_7/dialogue");
    }
}

TEST(Logging, Log_base_logger_is_unchanged_by_for_request) {
    CollectingLog logs;
    Chorus::Logger base = logs.logger();
    (void)base.for_request(42, std::string("npc_7"));

    base.warn("Engine-wide");

    ASSERT_TRUE(!logs.records()[0].request_id.has_value());
    ASSERT_TRUE(!logs.records()[0].session_id.has_value());
}

TEST(Logging, Log_request_scoped_logger_may_name_no_session) {
    CollectingLog logs;
    Chorus::Logger scoped = logs.logger().for_request(7);

    scoped.warn("Stateless");

    ASSERT_EQ(*logs.records()[0].request_id, int64_t{7});
    ASSERT_TRUE(!logs.records()[0].session_id.has_value());
}

// ---------------------------------------------------------------------------
// Timestamps
// ---------------------------------------------------------------------------

Chorus::LogRecord numbered_record(int64_t n) {
    Chorus::LogRecord record;
    record.message = "Record";
    record.fields = {{"n", n}};
    return record;
}

// A record is stamped where it is produced. A host collects later, on its own
// thread at its own cadence, so a stamp taken at collection would report when
// someone looked rather than when anything happened.
TEST(Logging, Log_record_is_stamped_when_produced) {
    auto channel = std::make_shared<Chorus::LogChannel>(8);
    Chorus::Logger log(channel, Chorus::LogLevel::Debug);

    const auto before = std::chrono::system_clock::now();
    log.warn("Produced");
    const auto after = std::chrono::system_clock::now();

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto drained = channel->drain();

    ASSERT_EQ(drained.size(), size_t{1});
    ASSERT_TRUE(drained[0].timestamp >= before);
    ASSERT_TRUE(drained[0].timestamp <= after); // and so well before the drain
}

// The report is batch metadata prefixed to the surviving records. Its timestamp
// says when loss began, independent of the records' production ordering.
TEST(Logging, Log_drop_report_stamped_with_first_loss) {
    Chorus::LogChannel channel(1);
    auto stamped = [](int64_t n, std::chrono::system_clock::time_point at) {
        Chorus::LogRecord record = numbered_record(n);
        record.timestamp = at;
        return record;
    };

    const auto start = std::chrono::system_clock::now();
    channel.push(stamped(1, start));
    const auto before_loss = std::chrono::system_clock::now();
    channel.push(stamped(2, start)); // evicts 1: the burst begins here
    const auto after_loss = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto much_later = std::chrono::system_clock::now();
    channel.push(stamped(3, much_later));

    auto drained = channel.drain();
    ASSERT_EQ(drained.size(), size_t{2});
    ASSERT_EQ(drained[0].message, "Log records dropped");
    ASSERT_TRUE(drained[0].timestamp >= before_loss);
    ASSERT_TRUE(drained[0].timestamp <= after_loss);
    ASSERT_TRUE(drained[0].timestamp < much_later);
}

// ---------------------------------------------------------------------------
// The channel
// ---------------------------------------------------------------------------

TEST(Logging, Log_channel_drains_in_order_and_empties) {
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
TEST(Logging, Log_channel_overflow_drops_oldest_and_reports) {
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
TEST(Logging, Log_channel_drop_count_resets_after_it_surfaces) {
    Chorus::LogChannel channel(1);
    channel.push(numbered_record(1));
    channel.push(numbered_record(2));
    ASSERT_EQ(channel.drain().size(), size_t{2});

    channel.push(numbered_record(3));
    auto drained = channel.drain();
    ASSERT_EQ(drained.size(), size_t{1});
    ASSERT_EQ(drained[0].message, "Record");
}

// Severity does not change queue order. After a stalled host, the newest window
// is more useful than an old warning followed by a hole in the stream.
TEST(Logging, Log_channel_overflow_drops_oldest_regardless_of_level) {
    Chorus::LogChannel channel(3);
    auto at_level = [](Chorus::LogLevel level, int64_t n) {
        Chorus::LogRecord record = numbered_record(n);
        record.level = level;
        return record;
    };

    channel.push(at_level(Chorus::LogLevel::Warn, 1));
    channel.push(at_level(Chorus::LogLevel::Info, 2));
    channel.push(at_level(Chorus::LogLevel::Info, 3));
    channel.push(at_level(Chorus::LogLevel::Info, 4));

    auto drained = channel.drain();
    ASSERT_EQ(drained.size(), size_t{4}); // one drop report plus three survivors
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[1], "n")), int64_t{2});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[2], "n")), int64_t{3});
    ASSERT_EQ(std::get<int64_t>(*find_log_field(drained[3], "n")), int64_t{4});
}

TEST(Logging, Log_channel_survives_many_producer_threads) {
    auto channel = std::make_shared<Chorus::LogChannel>(4096);
    Chorus::Logger log(channel, Chorus::LogLevel::Debug);

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
