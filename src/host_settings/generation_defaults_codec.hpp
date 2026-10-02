#pragma once

#include "chorus/runtime/runtime.hpp"

#include <string>
#include <string_view>

namespace chorus_host_settings {

struct ParseResult {
    Chorus::GenerationDefaults defaults;
    std::string path;
    std::string error;
    bool ok() const { return error.empty(); }
};

struct SerializeResult {
    std::string json;
    std::string path;
    std::string error;
    bool ok() const { return error.empty(); }
};

ParseResult parse_generation_defaults(std::string_view bytes);
SerializeResult serialize_generation_defaults(const Chorus::GenerationDefaults& defaults);

} // namespace chorus_host_settings
