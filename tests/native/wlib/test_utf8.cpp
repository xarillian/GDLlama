#include "gtest_utils.hpp"
#include "wlib/utf8.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>

namespace {

std::string_view bytes(const char* data, std::size_t size) {
    return {data, size};
}

TEST(Utf8, wLib_UTF_8_accepts_empty_and_ASCII) {
    ASSERT_EQ(wlib::valid_utf8_prefix_length(""), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length("plain ASCII"), 11u);
}

TEST(Utf8, wLib_UTF_8_accepts_multibyte_scalars) {
    const std::string text("A\xC2\xA2\xE2\x82\xAC\xF0\x9F\xA6\x8B", 10);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(text), text.size());
}

TEST(Utf8, wLib_UTF_8_accepts_scalar_boundaries) {
    const std::string text("\xE0\xA0\x80\xED\x9F\xBF\xF0\x90\x80\x80\xF4\x8F\xBF\xBF", 14);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(text), text.size());
}

TEST(Utf8, wLib_UTF_8_stops_before_incomplete_sequence) {
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xC2", 1)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xE2\x82", 2)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xF0\x9F\xA6", 3)), 0u);
}

TEST(Utf8, wLib_UTF_8_rejects_invalid_byte_structure) {
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\x80", 1)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xC0\x80", 2)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xF5\x80\x80\x80", 4)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xE2\x28\xA1", 3)), 0u);
}

TEST(Utf8, wLib_UTF_8_rejects_overlong_sequences) {
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xE0\x9F\x80", 3)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xF0\x8F\xBF\xBF", 4)), 0u);
}

TEST(Utf8, wLib_UTF_8_rejects_non_scalar_values) {
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xED\xA0\x80", 3)), 0u);
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("\xF4\x90\x80\x80", 4)), 0u);
}

TEST(Utf8, wLib_UTF_8_returns_valid_length_before_invalid_input) {
    ASSERT_EQ(wlib::valid_utf8_prefix_length(bytes("ok\xE2\x28\xA1", 5)), 2u);
}

} // namespace
