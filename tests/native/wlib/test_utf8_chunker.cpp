#include "gtest_utils.hpp"
#include "wlib/utf8.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

TEST(Utf8Chunker, wLib_Utf8Chunker_passes_ASCII_through) {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("hello"), std::string("hello"));
    ASSERT_EQ(chunker.push(""), std::string(""));
}

struct SplitScalarCase {
    const char* name;
    std::vector<std::string> pieces;
    std::string scalar;
};

class Utf8ChunkerSplitScalar : public ::testing::TestWithParam<SplitScalarCase> {};

TEST_P(Utf8ChunkerSplitScalar, Holds_incomplete_tail_until_scalar_is_complete) {
    const auto& test = GetParam();
    wlib::Utf8Chunker chunker;

    for (size_t index = 0; index + 1 < test.pieces.size(); ++index)
        ASSERT_EQ(chunker.push(test.pieces[index]), std::string{}) << test.name << " piece " << index;

    ASSERT_EQ(chunker.push(test.pieces.back()), test.scalar) << test.name;
    ASSERT_EQ(chunker.push(""), std::string{}) << test.name << " emitted more than once";
}

INSTANTIATE_TEST_SUITE_P(
    LegalMultibyteWidths,
    Utf8ChunkerSplitScalar,
    ::testing::Values(
        SplitScalarCase{"TwoByte", {"\xC2", "\xA2"}, "\xC2\xA2"},
        SplitScalarCase{"ThreeByte", {"\xE2", "\x82", "\xAC"}, "\xE2\x82\xAC"},
        SplitScalarCase{"FourByte", {"\xF0\x9F", "\xA6\x8B"}, "\xF0\x9F\xA6\x8B"}
    ),
    [](const ::testing::TestParamInfo<SplitScalarCase>& info) { return info.param.name; }
);

TEST(Utf8Chunker, wLib_Utf8Chunker_releases_valid_prefix_before_tail) {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("ok\xE2\x82"), std::string("ok"));
    ASSERT_EQ(chunker.push("\xACgo"), std::string("\xE2\x82\xACgo"));
}

TEST(Utf8Chunker, wLib_Utf8Chunker_drops_malformed_bytes_without_damming) {
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

TEST(Utf8Chunker, wLib_Utf8Chunker_drops_overlong_when_disproven) {
    wlib::Utf8Chunker chunker;
    // E0 9F would be an overlong encoding: rejected as soon as seen together.
    ASSERT_EQ(chunker.push("\xE0"), std::string(""));
    ASSERT_EQ(chunker.push("\x9F\x80"), std::string(""));
    ASSERT_EQ(chunker.push("ok"), std::string("ok"));
}

TEST(Utf8Chunker, wLib_Utf8Chunker_replace_policy_keeps_corruption_visible) {
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

TEST(Utf8Chunker, wLib_Utf8Chunker_reset_discards_held_tail) {
    wlib::Utf8Chunker chunker;
    ASSERT_EQ(chunker.push("\xE2\x82"), std::string(""));
    chunker.reset();
    // The freed continuation byte is malformed on its own and drops.
    ASSERT_EQ(chunker.push("\xAC"), std::string(""));
    ASSERT_EQ(chunker.push("fresh"), std::string("fresh"));
}

} // namespace
