#include "gtest_filter.hpp"

#include <gtest/gtest.h>

namespace {

TEST(GtestSubstringFilterTest, Empty_fragment_keeps_only_outer_wildcards) {
    EXPECT_EQ(make_gtest_substring_filter(""), "**");
}

TEST(GtestSubstringFilterTest, Fragment_is_wrapped_without_rewriting) {
    EXPECT_EQ(
        make_gtest_substring_filter("Mixed Case.Suite/0-value"),
        "*Mixed Case.Suite/0-value*"
    );
}

} // namespace
