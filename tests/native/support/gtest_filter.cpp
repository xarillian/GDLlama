#include "gtest_filter.hpp"

std::string make_gtest_substring_filter(std::string_view substring) {
    std::string filter;
    filter.reserve(substring.size() + 2);
    filter += '*';
    filter += substring;
    filter += '*';
    return filter;
}
