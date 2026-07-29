#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace Chorus {

// JSON-compatible option value for namespaced provider options (spec 3c-B).
// Behavior-free, copyable, std-only. Recursion goes through std::vector
// (guaranteed to support incomplete types) and std::map (supported by
// libstdc++/libc++/MSVC in practice; nlohmann/json relies on the same
// property). Verified by compile spike on gcc + clang, 2026-07-12; the
// Windows build validates MSVC. Fallback if a platform balks: replace the
// map alternative with a sorted std::vector<std::pair<std::string, OptionValue>>.
struct OptionValue;
using OptionList = std::vector<OptionValue>;
using OptionMap = std::map<std::string, OptionValue>;

struct OptionValue : std::variant<bool, int64_t, double, std::string, OptionList, OptionMap> {
    using variant::variant;
    OptionValue(const char* s) : variant(std::string(s)) {}
};

} // namespace Chorus
