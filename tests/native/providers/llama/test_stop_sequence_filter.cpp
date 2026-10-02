#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "gtest_utils.hpp"
#include "wlib/utf8.hpp"

#include <iostream>
#include <string>
#include <vector>

TEST(StopSequenceFilter, Stop_filter_spans_pieces) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto a = filter.push("hello <E");
    ASSERT_EQ(a.safe_text, "hello ");
    ASSERT_TRUE(!a.matched);
    auto b = filter.push("ND>ignored");
    ASSERT_TRUE(b.matched);
    ASSERT_EQ(b.safe_text, "");
}

TEST(StopSequenceFilter, Stop_filter_holds_split_UTF_8) {
    Chorus::StopSequenceFilter filter({"<STOP>"});
    auto a = filter.push(std::string("A\xF0\x9F", 3));
    ASSERT_EQ(a.safe_text, "A");
    auto b = filter.push(std::string("\xA6\x8B", 2));
    ASSERT_EQ(b.safe_text, std::string("\xF0\x9F\xA6\x8B", 4));
}

TEST(StopSequenceFilter, Stop_filter_matches_in_one_piece) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push("hello<END>ignored");
    ASSERT_EQ(result.safe_text, "hello");
    ASSERT_TRUE(result.matched);
}

TEST(StopSequenceFilter, Stop_filter_matches_across_three_pieces) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto a = filter.push("hello<");
    auto b = filter.push("EN");
    auto c = filter.push("D>ignored");
    ASSERT_EQ(a.safe_text, "hello");
    ASSERT_EQ(b.safe_text, "");
    ASSERT_EQ(c.safe_text, "");
    ASSERT_TRUE(c.matched);
}

TEST(StopSequenceFilter, Stop_filter_preserves_overlapping_internal_prefixes) {
    Chorus::StopSequenceFilter filter({"ababx"});
    auto a = filter.push("abab");
    auto b = filter.push("abxignored");
    ASSERT_EQ(a.safe_text, "");
    ASSERT_EQ(b.safe_text, "ab");
    ASSERT_TRUE(b.matched);
}

TEST(StopSequenceFilter, Stop_filter_chooses_earliest_marker) {
    Chorus::StopSequenceFilter filter({"<LATE>", "<EARLY>"});
    auto result = filter.push("a<EARLY>b<LATE>");
    ASSERT_EQ(result.safe_text, "a");
    ASSERT_TRUE(result.matched);
}

TEST(StopSequenceFilter, Stop_filter_without_markers_emits_text) {
    Chorus::StopSequenceFilter filter({});
    auto result = filter.push("ordinary text");
    ASSERT_EQ(result.safe_text, "ordinary text");
    ASSERT_TRUE(!result.matched);
    ASSERT_EQ(filter.flush(), "");
}

TEST(StopSequenceFilter, Stop_filter_ordinary_flush) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push("hello");
    ASSERT_EQ(result.safe_text, "hello");
    ASSERT_EQ(filter.flush(), "");
}

TEST(StopSequenceFilter, Stop_filter_flushes_partial_marker) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push("hello <E");
    ASSERT_EQ(result.safe_text, "hello ");
    ASSERT_EQ(filter.flush(), "<E");
}

TEST(StopSequenceFilter, Stop_filter_matches_marker_in_final_piece) {
    Chorus::StopSequenceFilter filter({"END"});
    auto result = filter.finish("ENDvisible");
    ASSERT_TRUE(result.matched);
    ASSERT_TRUE(result.safe_text.empty());
}

TEST(StopSequenceFilter, Stop_filter_matches_marker_across_final_piece) {
    Chorus::StopSequenceFilter filter({"END"});
    auto prefix = filter.push("safe E");
    ASSERT_EQ(prefix.safe_text, std::string("safe "));
    auto result = filter.finish("NDvisible");
    ASSERT_TRUE(result.matched);
    ASSERT_TRUE(result.safe_text.empty());
}

TEST(StopSequenceFilter, Stop_filter_drops_incomplete_UTF_8_before_match) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto result = filter.push(std::string("A\xF0\x9F<END>", 8));
    ASSERT_EQ(result.safe_text, "A");
    ASSERT_TRUE(result.matched);
}

TEST(StopSequenceFilter, Stop_filter_drops_malformed_byte_without_damming) {
    Chorus::StopSequenceFilter filter({"<END>"});
    auto malformed = filter.push(std::string("\x80", 1));
    ASSERT_TRUE(malformed.safe_text.empty());
    ASSERT_TRUE(!malformed.matched);
    auto recovered = filter.push("abc");
    ASSERT_EQ(recovered.safe_text, std::string("abc"));
    ASSERT_TRUE(!recovered.matched);
    ASSERT_EQ(filter.flush(), std::string{});
}

TEST(StopSequenceFilter, Final_content_filter_does_not_synthesize_stop_marker_after_UTF_8_recovery) {
    Chorus::StopSequenceFilter filter({"AB"});
    wlib::Utf8Chunker content_chunker;
    std::string raw = "A";
    raw.push_back(static_cast<char>(0x80));
    raw += "B";

    auto result = Chorus::finish_content_stream(&filter, content_chunker, raw);

    ASSERT_TRUE(!result.matched);
    ASSERT_EQ(result.safe_text, std::string("AB"));
}

TEST(StopSequenceFilter, Final_content_without_stop_filter_drops_incomplete_UTF_8_tail) {
    wlib::Utf8Chunker content_chunker;
    const std::string raw("A\xF0", 2);

    auto result = Chorus::finish_content_stream(nullptr, content_chunker, raw);

    ASSERT_TRUE(!result.matched);
    ASSERT_EQ(result.safe_text, std::string("A"));
}

TEST(StopSequenceFilter, Stop_filter_rejects_identical_markers) {
    auto rejection = Chorus::validate_stop_sequences({"<END>", "<END>"});
    ASSERT_TRUE(rejection.has_value());
    ASSERT_TRUE(rejection->error == Chorus::ChorusError::InvalidRequest);
    ASSERT_TRUE(rejection->message.find("<END>") != std::string::npos);
    ASSERT_TRUE(rejection->message.find("duplicate") != std::string::npos);
}

TEST(StopSequenceFilter, Stop_filter_rejects_empty_marker) {
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

TEST(StopSequenceFilter, Stop_filter_rejects_prefix_markers_short_first) {
    assert_prefix_rejection_names_both({"<END", "<END_JSON>"});
}

TEST(StopSequenceFilter, Stop_filter_rejects_prefix_markers_long_first) {
    assert_prefix_rejection_names_both({"<END_JSON>", "<END"});
}

TEST(StopSequenceFilter, Stop_filter_accepts_markers_without_byte_prefix_relation) {
    auto rejection = Chorus::validate_stop_sequences({"<END>", "<END_JSON>"});
    ASSERT_TRUE(!rejection.has_value());
}
