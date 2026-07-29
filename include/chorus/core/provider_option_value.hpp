#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace Chorus {

/*
 * A JSON-shaped value carried in a provider's option map: a bool, integer,
 * double, or string, or a list or string-keyed map of further values.
 *
 * The list and map alternatives name `ProviderOptionValue` while it is still
 * incomplete.
 * `std::vector` guarantees support for incomplete element types,
 * `std::map` supplies it in practice on libstdc++, libc++, and MSVC.
 * Should a platform balk at the map, the fallback is a sorted
 * `std::vector<std::pair<std::string, ProviderOptionValue>>`.
 */
struct ProviderOptionValue;
using ProviderOptionList = std::vector<ProviderOptionValue>;
using ProviderOptionMap = std::map<std::string, ProviderOptionValue>;

struct ProviderOptionValue : std::variant<bool, int64_t, double, std::string, ProviderOptionList, ProviderOptionMap> {
    using variant::variant;

    // Routes string literals to the string alternative; without this they decay to bool.
    ProviderOptionValue(const char* s) : variant(std::string(s)) {}
};

} // namespace Chorus
