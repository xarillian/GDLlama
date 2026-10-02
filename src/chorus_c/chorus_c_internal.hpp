#pragma once

#include "chorus/runtime/runtime.hpp"
#include "chorus_c/chorus_c.h"

#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct chorus_request {
    Chorus::GenerationRequest value;
};

struct chorus_runtime {
    Chorus::ChorusRuntime value;
    Chorus::GenerationDefaults generation_defaults;
    mutable std::string last_error;

    std::vector<Chorus::RuntimeEvent> event_source;
    std::vector<chorus_event> events;

    std::deque<std::string> result_strings;

    std::vector<Chorus::LogRecord> log_source;
    std::vector<std::vector<chorus_log_field>> log_fields;
    std::vector<chorus_log_record> logs;
};

namespace chorus_c {

std::optional<Chorus::MessageRole> to_cpp_role(chorus_message_role role) noexcept;
const char* error_name(chorus_error error) noexcept;
chorus_error to_c_error(Chorus::ChorusError error) noexcept;
bool to_cpp_execution_mode(chorus_execution_mode mode, Chorus::ExecutionMode& out) noexcept;

void replace_last_error(const chorus_runtime* rt, std::string_view detail) noexcept;
void clear_last_error(const chorus_runtime* rt) noexcept;
chorus_error invalid_request(const chorus_runtime* rt, std::string_view detail) noexcept;
chorus_error unknown_exception(const chorus_runtime* rt, const char* detail) noexcept;

char* copy_owned_string(std::string_view value) noexcept;
void clear_result_storage(chorus_runtime* rt) noexcept;

template <typename Action> chorus_error guard_builder(Action&& action) noexcept {
    try {
        action();
        return CHORUS_OK;
    } catch (...) {
        return CHORUS_ERR_UNKNOWN;
    }
}

} // namespace chorus_c
