#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace Chorus {

/*
 * A JSON-shaped provider option value: a bool, integer, double, string,
 * list, or string-keyed map.
 */
struct ProviderOptionValue;
using ProviderOptionList = std::vector<ProviderOptionValue>;
using ProviderOptionMap = std::map<std::string, ProviderOptionValue>;

struct ProviderOptionValue : std::variant<bool, int64_t, double, std::string, ProviderOptionList, ProviderOptionMap> {
    using variant::variant;

    // Routes string literals to `std::string`; otherwise they convert to `bool`.
    ProviderOptionValue(const char* s) : variant(std::string(s)) {}
};

} // namespace Chorus
