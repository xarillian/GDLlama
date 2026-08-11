#include "chorus/providers/llama/llama_log_bridge.hpp"
#include "collecting_sink.hpp"
#include "test_utils.hpp"

#include "llama.h"

#include <iostream>
#include <string>

// The assembler is the bridge's whole adaptation layer, and it is testable
// without touching llama's process-global hook. No model needed: these run
// under CHORUS_SKIP_MODEL_TESTS=1.

namespace {

// llama writes a line in pieces and terminates it with its own newline.
void test_a_single_terminated_fragment_becomes_one_record() {
    Chorus::LlamaLogAssembler assembler;

    auto records = assembler.feed(GGML_LOG_LEVEL_INFO, "load: loaded 24 tensors\n");

    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_EQ(records[0].message, "load: loaded 24 tensors");
    ASSERT_TRUE(records[0].level == Chorus::LogLevel::Info);
}

void test_one_callback_with_two_lines_becomes_two_records() {
    Chorus::LlamaLogAssembler assembler;

    auto records = assembler.feed(GGML_LOG_LEVEL_INFO, "first line\nsecond line\n");

    ASSERT_EQ(records.size(), size_t{2});
    ASSERT_EQ(records[0].message, "first line");
    ASSERT_EQ(records[1].message, "second line");
}

void test_cont_fragments_join_into_one_record() {
    Chorus::LlamaLogAssembler assembler;

    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_WARN, "load: ").empty());
    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_CONT, "control token ").empty());
    auto records = assembler.feed(GGML_LOG_LEVEL_CONT, "106 is not marked as EOG\n");

    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_EQ(records[0].message, "load: control token 106 is not marked as EOG");
    // The level of the fragment that opened the line rules; CONT carries none.
    ASSERT_TRUE(records[0].level == Chorus::LogLevel::Warn);
}

// llama leaving a line unterminated must not swallow the next fragment's
// level: an error merged into an open Info line is an error a mask can hide.
void test_a_new_opener_closes_a_stale_line_and_keeps_its_own_level() {
    Chorus::LlamaLogAssembler assembler;

    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_INFO, "load: chatty progress").empty());
    auto records = assembler.feed(GGML_LOG_LEVEL_ERROR, "failed to allocate buffer\n");

    ASSERT_EQ(records.size(), size_t{2});
    ASSERT_EQ(records[0].message, "load: chatty progress"); // the stale line, as it stood
    ASSERT_TRUE(records[0].level == Chorus::LogLevel::Info);
    ASSERT_EQ(records[1].message, "failed to allocate buffer");
    ASSERT_TRUE(records[1].level == Chorus::LogLevel::Error);
}

void test_trailing_newlines_are_trimmed() {
    Chorus::LlamaLogAssembler assembler;

    auto records = assembler.feed(GGML_LOG_LEVEL_ERROR, "failed to load\r\n");

    ASSERT_EQ(records.size(), size_t{1});
    ASSERT_EQ(records[0].message, "failed to load");
}

// Every level llama means for a reader maps across; NONE drops.
void test_level_mapping_is_exhaustive() {
    struct Case {
        int ggml_level;
        Chorus::LogLevel expected;
    };
    const Case cases[] = {
        {GGML_LOG_LEVEL_DEBUG, Chorus::LogLevel::Debug},
        {GGML_LOG_LEVEL_INFO, Chorus::LogLevel::Info},
        {GGML_LOG_LEVEL_WARN, Chorus::LogLevel::Warn},
        {GGML_LOG_LEVEL_ERROR, Chorus::LogLevel::Error},
    };
    for (const auto& c : cases) {
        Chorus::LlamaLogAssembler assembler;
        auto records = assembler.feed(c.ggml_level, "line\n");
        ASSERT_EQ(records.size(), size_t{1});
        ASSERT_TRUE(records[0].level == c.expected);
    }

    Chorus::LlamaLogAssembler none_assembler;
    ASSERT_TRUE(none_assembler.feed(GGML_LOG_LEVEL_NONE, "line\n").empty());
}

void test_an_empty_or_null_fragment_is_ignored() {
    Chorus::LlamaLogAssembler assembler;
    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_INFO, nullptr).empty());
    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_INFO, "").empty());
}

// A continuation with nothing open is llama's state, not ours, to be confused
// about; the record it would open has no level, so it is dropped.
void test_a_continuation_of_nothing_is_dropped() {
    Chorus::LlamaLogAssembler assembler;
    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_CONT, "orphan\n").empty());
}

void test_flush_emits_an_unterminated_line() {
    Chorus::LlamaLogAssembler assembler;
    ASSERT_TRUE(assembler.feed(GGML_LOG_LEVEL_INFO, "no newline here").empty());

    auto record = assembler.flush();
    ASSERT_TRUE(record.has_value());
    ASSERT_EQ(record->message, "no newline here");
    ASSERT_TRUE(!assembler.flush().has_value()); // and forgets it
}

// The end-to-end path: a registered logger sees llama's own output, and stops
// seeing it once the handle is gone.
void test_the_bridge_routes_llama_output_and_restores_the_hook_on_release() {
    ggml_log_callback previous_callback = nullptr;
    void* previous_user_data = nullptr;
    llama_log_get(&previous_callback, &previous_user_data);

    CollectingSink sink;
    {
        auto bridge = Chorus::LlamaLogBridge::acquire(sink.logger());

        ggml_log_callback installed = nullptr;
        void* installed_user_data = nullptr;
        llama_log_get(&installed, &installed_user_data);
        ASSERT_TRUE(installed != previous_callback);

        installed(GGML_LOG_LEVEL_WARN, "vendor said something\n", installed_user_data);

        ASSERT_EQ(sink.size(), size_t{1});
        ASSERT_EQ(sink.records()[0].message, "vendor said something");
    }

    ggml_log_callback restored = nullptr;
    void* restored_user_data = nullptr;
    llama_log_get(&restored, &restored_user_data);
    ASSERT_TRUE(restored == previous_callback);
    ASSERT_TRUE(restored_user_data == previous_user_data);
}

// Two live engines are two registrations and one hook. Attribution is not
// available from llama, so both hear everything, on purpose.
void test_two_registrations_both_receive_vendor_output() {
    CollectingSink first;
    CollectingSink second;
    auto bridge_a = Chorus::LlamaLogBridge::acquire(first.logger());
    auto bridge_b = Chorus::LlamaLogBridge::acquire(second.logger());

    ggml_log_callback installed = nullptr;
    void* user_data = nullptr;
    llama_log_get(&installed, &user_data);
    installed(GGML_LOG_LEVEL_INFO, "shared line\n", user_data);

    ASSERT_EQ(first.size(), size_t{1});
    ASSERT_EQ(second.size(), size_t{1});

    // The hook survives one departure and only lifts with the last.
    bridge_a.reset();
    installed(GGML_LOG_LEVEL_INFO, "after one left\n", user_data);
    ASSERT_EQ(first.size(), size_t{1});
    ASSERT_EQ(second.size(), size_t{2});
}

} // namespace

int run_llama_log_bridge_tests() {
    std::cout << "\n--- Llama Log Bridge Tests ---\n";
    run_test(
        "Llama log single terminated fragment becomes one record", test_a_single_terminated_fragment_becomes_one_record
    );
    run_test("Llama log callback splits physical lines", test_one_callback_with_two_lines_becomes_two_records);
    run_test("Llama log CONT fragments join into one record", test_cont_fragments_join_into_one_record);
    run_test(
        "Llama log new opener closes a stale line and keeps its level",
        test_a_new_opener_closes_a_stale_line_and_keeps_its_own_level
    );
    run_test("Llama log trailing newlines are trimmed", test_trailing_newlines_are_trimmed);
    run_test("Llama log level mapping is exhaustive", test_level_mapping_is_exhaustive);
    run_test("Llama log empty or null fragment is ignored", test_an_empty_or_null_fragment_is_ignored);
    run_test("Llama log continuation of nothing is dropped", test_a_continuation_of_nothing_is_dropped);
    run_test("Llama log flush emits an unterminated line", test_flush_emits_an_unterminated_line);
    run_test(
        "Llama log bridge routes output and restores the hook",
        test_the_bridge_routes_llama_output_and_restores_the_hook_on_release
    );
    run_test("Llama log bridge multiplexes to every registration", test_two_registrations_both_receive_vendor_output);
    return g_tests_failed;
}
