#include "test_utils.hpp"
#include "wlib/utf8.hpp"

#include <iostream>
#include <string>

namespace {

void test_chunker_passes_ascii_through() {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("hello"), std::string("hello"));
    ASSERT_EQ(chunker.push(""), std::string(""));
}

void test_chunker_holds_split_two_byte_sequence() {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("\xC2"), std::string(""));
    ASSERT_EQ(chunker.push("\xA2"), std::string("\xC2\xA2"));
}

void test_chunker_holds_three_byte_sequence_across_three_pushes() {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("\xE2"), std::string(""));
    ASSERT_EQ(chunker.push("\x82"), std::string(""));
    ASSERT_EQ(chunker.push("\xAC"), std::string("\xE2\x82\xAC"));
}

void test_chunker_holds_split_four_byte_sequence() {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("\xF0\x9F"), std::string(""));
    ASSERT_EQ(chunker.push("\xA6\x8B"), std::string("\xF0\x9F\xA6\x8B"));
}

void test_chunker_releases_valid_prefix_before_held_tail() {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("ok\xE2\x82"), std::string("ok"));
    ASSERT_EQ(chunker.push("\xACgo"), std::string("\xE2\x82\xACgo"));
}

void test_chunker_drops_malformed_bytes_without_damming() {
    wlib::Utf8Chunker chunker;
    // A stray continuation byte must not block what follows it.
    ASSERT_EQ(
        chunker.push(
            "\x80"
            "abc"
        ),
        std::string("abc")
    );
    // A lead byte orphaned by a second lead: the orphan drops, the pair emits.
    wlib::Utf8Chunker chunker2;
    ASSERT_EQ(chunker2.push("\xC2\xC2\xA2"), std::string("\xC2\xA2"));
    // An invalid lead never enters pending.
    wlib::Utf8Chunker chunker3;
    ASSERT_EQ(chunker3.push("\xF5"), std::string(""));
    ASSERT_EQ(chunker3.push("ok"), std::string("ok"));
}

void test_chunker_drops_overlong_when_disproven() {
    wlib::Utf8Chunker chunker;
    // E0 9F would be an overlong encoding: rejected as soon as seen together.
    ASSERT_EQ(chunker.push("\xE0"), std::string(""));
    ASSERT_EQ(chunker.push("\x9F\x80"), std::string(""));
    ASSERT_EQ(chunker.push("ok"), std::string("ok"));
}

void test_chunker_replace_policy_keeps_corruption_visible() {
    // Replace substitutes one U+FFFD per rejected byte instead of silence.
    wlib::Utf8Chunker chunker{wlib::Utf8InvalidBytePolicy::Replace};
    ASSERT_EQ(
        chunker.push(
            "\x80"
            "abc"
        ),
        std::string(
            "\xEF\xBF\xBD"
            "abc"
        )
    );
    // An orphaned lead is replaced; the valid pair behind it still emits.
    wlib::Utf8Chunker chunker2{wlib::Utf8InvalidBytePolicy::Replace};
    ASSERT_EQ(chunker2.push("\xC2\xC2\xA2"), std::string("\xEF\xBF\xBD\xC2\xA2"));
    // Merely-split sequences are held and completed, never replaced.
    wlib::Utf8Chunker chunker3{wlib::Utf8InvalidBytePolicy::Replace};
    ASSERT_EQ(chunker3.push("\xE2\x82"), std::string(""));
    ASSERT_EQ(chunker3.push("\xAC"), std::string("\xE2\x82\xAC"));
}

void test_chunker_reset_discards_held_tail() {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("\xE2\x82"), std::string(""));
    chunker.reset();
    // The freed continuation byte is malformed on its own and drops.
    ASSERT_EQ(chunker.push("\xAC"), std::string(""));
    ASSERT_EQ(chunker.push("fresh"), std::string("fresh"));
}

} // namespace

int run_wlib_utf8_chunker_tests() {
    std::cout << "\n=== WLIB UTF-8 CHUNKER SUITE ===\n";

    run_test("wLib Utf8Chunker: passes ASCII through", test_chunker_passes_ascii_through);
    run_test("wLib Utf8Chunker: holds split two-byte sequence", test_chunker_holds_split_two_byte_sequence);
    run_test(
        "wLib Utf8Chunker: holds three-byte sequence across pushes",
        test_chunker_holds_three_byte_sequence_across_three_pushes
    );
    run_test("wLib Utf8Chunker: holds split four-byte sequence", test_chunker_holds_split_four_byte_sequence);
    run_test(
        "wLib Utf8Chunker: releases valid prefix before tail", test_chunker_releases_valid_prefix_before_held_tail
    );
    run_test(
        "wLib Utf8Chunker: drops malformed bytes without damming", test_chunker_drops_malformed_bytes_without_damming
    );
    run_test("wLib Utf8Chunker: drops overlong when disproven", test_chunker_drops_overlong_when_disproven);
    run_test(
        "wLib Utf8Chunker: replace policy keeps corruption visible",
        test_chunker_replace_policy_keeps_corruption_visible
    );
    run_test("wLib Utf8Chunker: reset discards held tail", test_chunker_reset_discards_held_tail);

    return g_tests_failed > 0 ? 1 : 0;
}
