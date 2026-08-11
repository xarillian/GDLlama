#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "test_utils.hpp"
#include "wlib/utf8.hpp"

#include <iostream>
#include <string>
#include <vector>

void test_stop_filter_spanning_pieces() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto a = filter.push("hello <E");
    ASSERT_EQ(a.safe_text, "hello ");
    ASSERT_TRUE(!a.matched);
    auto b = filter.push("ND>ignored");
    ASSERT_TRUE(b.matched);
    ASSERT_EQ(b.safe_text, "");
}

void test_stop_filter_holds_split_utf8() {
    Chorus::StopSequenceFilter filter({"<STOP>"});
    auto a = filter.push(std::string("A\xF0\x9F", 3));
    ASSERT_EQ(a.safe_text, "A");
    auto b = filter.push(std::string("\xA6\x8B", 2));
    ASSERT_EQ(b.safe_text, std::string("\xF0\x9F\xA6\x8B", 4));
}

void test_stop_filter_matches_in_one_piece() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push("hello<END>ignored");
    ASSERT_EQ(result.safe_text, "hello");
    ASSERT_TRUE(result.matched);
}

void test_stop_filter_matches_across_three_pieces() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto a = filter.push("hello<");
    auto b = filter.push("EN");
    auto c = filter.push("D>ignored");
    ASSERT_EQ(a.safe_text, "hello");
    ASSERT_EQ(b.safe_text, "");
    ASSERT_EQ(c.safe_text, "");
    ASSERT_TRUE(c.matched);
}

void test_stop_filter_preserves_overlapping_internal_prefixes() {
    Chorus::StopSequenceFilter filter({"ababx"});
    auto a = filter.push("abab");
    auto b = filter.push("abxignored");
    ASSERT_EQ(a.safe_text, "");
    ASSERT_EQ(b.safe_text, "ab");
    ASSERT_TRUE(b.matched);
}

void test_stop_filter_chooses_earliest_of_multiple_markers() {
    Chorus::StopSequenceFilter filter({"<LATE>", "<EARLY>"});
    auto result = filter.push("a<EARLY>b<LATE>");
    ASSERT_EQ(result.safe_text, "a");
    ASSERT_TRUE(result.matched);
}

void test_stop_filter_without_markers_emits_valid_text() {
    Chorus::StopSequenceFilter filter({});
    auto result = filter.push("ordinary text");
    ASSERT_EQ(result.safe_text, "ordinary text");
    ASSERT_TRUE(!result.matched);
    ASSERT_EQ(filter.flush(), "");
}

void test_stop_filter_flushes_ordinary_pending_text() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push("hello");
    ASSERT_EQ(result.safe_text, "hello");
    ASSERT_EQ(filter.flush(), "");
}

void test_stop_filter_flushes_partial_marker() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push("hello <E");
    ASSERT_EQ(result.safe_text, "hello ");
    ASSERT_EQ(filter.flush(), "<E");
}

void test_stop_filter_matches_marker_in_final_piece() {
    Chorus::StopSequenceFilter filter({"END"});
    auto result = filter.finish("ENDvisible");
    ASSERT_TRUE(result.matched);
    ASSERT_TRUE(result.safe_text.empty());
}

void test_stop_filter_matches_marker_across_final_piece() {
    Chorus::StopSequenceFilter filter({"END"});
    auto prefix = filter.push("safe E");
    ASSERT_EQ(prefix.safe_text, std::string("safe "));
    auto result = filter.finish("NDvisible");
    ASSERT_TRUE(result.matched);
    ASSERT_TRUE(result.safe_text.empty());
}

void test_stop_filter_drops_incomplete_utf8_before_match() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push(std::string("A\xF0\x9F<END>", 8));
    ASSERT_EQ(result.safe_text, "A");
    ASSERT_TRUE(result.matched);
}

void test_stop_filter_drops_malformed_byte_without_damming() {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto malformed = filter.push(std::string("\x80", 1));
    ASSERT_TRUE(malformed.safe_text.empty());
    ASSERT_TRUE(!malformed.matched);
    auto recovered = filter.push("abc");
    ASSERT_EQ(recovered.safe_text, std::string("abc"));
    ASSERT_TRUE(!recovered.matched);
    ASSERT_EQ(filter.flush(), std::string{});
}

void test_final_content_filter_does_not_match_marker_joined_by_utf8_recovery() {
    Chorus::StopSequenceFilter filter({"AB"});
    wlib::Utf8Chunker content_chunker;
    std::string raw = "A";
    raw.push_back(static_cast<char>(0x80));
    raw += "B";

    auto result = Chorus::finish_content_stream(&filter, content_chunker, raw);

    ASSERT_TRUE(!result.matched);
    ASSERT_EQ(result.safe_text, std::string("AB"));
}

void test_final_content_without_stop_filter_drops_incomplete_utf8_tail() {
    wlib::Utf8Chunker content_chunker;
    const std::string raw("A\xF0", 2);

    auto result = Chorus::finish_content_stream(nullptr, content_chunker, raw);

    ASSERT_TRUE(!result.matched);
    ASSERT_EQ(result.safe_text, std::string("A"));
}

void test_stop_filter_rejects_identical_markers() {
    auto rejection = Chorus::validate_stop_sequences({"<END>", "<END>"});
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(rejection->message.find("<END>") != std::string::npos);
    ASSERT_TRUE(rejection->message.find("duplicate") != std::string::npos);
}

void test_stop_filter_rejects_empty_marker() {
    auto rejection = Chorus::validate_stop_sequences({"<END>", ""});
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(rejection->message.find("empty") != std::string::npos);
}

void assert_prefix_rejection_names_both(const std::vector<std::string>& markers) {
    auto rejection = Chorus::validate_stop_sequences(markers);
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::InvalidRequest);
    ASSERT_EQ(
        rejection->message,
        std::string("Stop markers '") + markers[0] + "' and '" + markers[1] + "' are prefix-related."
    );
}

void test_stop_filter_rejects_prefix_markers_short_first() {
    assert_prefix_rejection_names_both({"<END", "<END_JSON>"});
}

void test_stop_filter_rejects_prefix_markers_long_first() {
    assert_prefix_rejection_names_both({"<END_JSON>", "<END"});
}

void test_stop_filter_accepts_markers_without_byte_prefix_relation() {
    auto rejection = Chorus::validate_stop_sequences({"<END>", "<END_JSON>"});
    ASSERT_TRUE(!rejection.has_value());
}

int run_stop_sequence_filter_tests() {
    std::cout << "\n=== STOP FILTER SUITE ===\n";

    run_test("Stop filter spans pieces", test_stop_filter_spanning_pieces);
    run_test("Stop filter holds split UTF-8", test_stop_filter_holds_split_utf8);
    run_test("Stop filter matches in one piece", test_stop_filter_matches_in_one_piece);
    run_test("Stop filter matches across three pieces", test_stop_filter_matches_across_three_pieces);
    run_test(
        "Stop filter preserves overlapping internal prefixes", test_stop_filter_preserves_overlapping_internal_prefixes
    );
    run_test("Stop filter chooses earliest marker", test_stop_filter_chooses_earliest_of_multiple_markers);
    run_test("Stop filter without markers emits text", test_stop_filter_without_markers_emits_valid_text);
    run_test("Stop filter ordinary flush", test_stop_filter_flushes_ordinary_pending_text);
    run_test("Stop filter flushes partial marker", test_stop_filter_flushes_partial_marker);
    run_test("Stop filter matches marker in final piece", test_stop_filter_matches_marker_in_final_piece);
    run_test("Stop filter matches marker across final piece", test_stop_filter_matches_marker_across_final_piece);
    run_test("Stop filter drops incomplete UTF-8 before match", test_stop_filter_drops_incomplete_utf8_before_match);
    run_test("Stop filter drops malformed byte without damming", test_stop_filter_drops_malformed_byte_without_damming);
    run_test(
        "Final content filter does not synthesize stop marker after UTF-8 recovery",
        test_final_content_filter_does_not_match_marker_joined_by_utf8_recovery
    );
    run_test(
        "Final content without stop filter drops incomplete UTF-8 tail",
        test_final_content_without_stop_filter_drops_incomplete_utf8_tail
    );
    run_test("Stop filter rejects identical markers", test_stop_filter_rejects_identical_markers);
    run_test("Stop filter rejects empty marker", test_stop_filter_rejects_empty_marker);
    run_test("Stop filter rejects prefix markers short first", test_stop_filter_rejects_prefix_markers_short_first);
    run_test("Stop filter rejects prefix markers long first", test_stop_filter_rejects_prefix_markers_long_first);
    run_test(
        "Stop filter accepts markers without byte prefix relation",
        test_stop_filter_accepts_markers_without_byte_prefix_relation
    );

    return g_tests_failed > 0 ? 1 : 0;
}
