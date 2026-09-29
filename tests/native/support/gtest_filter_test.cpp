#include "gtest_filter.hpp"

#include <gtest/gtest.h>

namespace {

TEST(GtestSubstringFilterTest, Fragment_is_wrapped_without_rewriting) {
    EXPECT_EQ(
        make_gtest_substring_filter("Mixed Case.Suite/0-value"),
        "*Mixed Case.Suite/0-value*"
    );
}

} // namespace
