#include "godot_chorus/project_generation_defaults.hpp"

#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string_view>
#include <system_error>
#include <variant>

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "godot_chorus/option_conversion.hpp"
#include "host_settings/generation_defaults_codec.hpp"

// After godot-cpp: Windows headers define macros that collide with Godot's generated names.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace godot;
namespace godot_chorus {
namespace {

constexpr const char* selector = "chorus/generation/settings_path";
constexpr const char* prefix = "chorus/generation/";
constexpr const char* fields[] = {
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "show_thinking",
    "constraint",
    "chat_template",
    "provider_options"
};

struct ProjectDefaults {
    Chorus::GenerationDefaults value;
    String path;
    bool failed = true;
    std::string imported_bytes;
};
std::optional<ProjectDefaults> state;

std::string text(const String& value) {
    const CharString bytes = value.utf8();
    return std::string(bytes.get_data(), static_cast<size_t>(bytes.length()));
}
std::filesystem::path utf8_path(const String& value) {
    const std::string bytes = text(value);
    return std::filesystem::path(std::u8string(bytes.begin(), bytes.end()));
}
String setting_name(const char* field) {
    return String(prefix) + field;
}

bool contained_path(const String& path, std::filesystem::path& loose, std::string& error) {
    if (!path.begins_with("res://")) {
        error = "settings_path must select a project-contained res:// file.";
        return false;
    }
    const String relative = path.substr(6);
    const bool drive_path = relative.length() >= 2 && relative[1] == ':' &&
                            ((relative[0] >= 'A' && relative[0] <= 'Z') || (relative[0] >= 'a' && relative[0] <= 'z'));
    if (relative.is_empty() || relative.is_absolute_path() || relative.find("\\") != -1 || drive_path ||
        relative.find("..") != -1) {
        error = "settings_path must select a project-contained res:// file.";
        return false;
    }
    const auto* settings = ProjectSettings::get_singleton();
    const String global_root = settings->globalize_path("res://");
    if (global_root.is_empty()) {
        if (settings->globalize_path(path).is_absolute_path()) {
            error = "settings_path escapes the project root.";
            return false;
        }
        loose.clear();
        return true;
    }
    std::error_code ec;
    const auto root = std::filesystem::weakly_canonical(utf8_path(global_root), ec);
    if (ec) {
        error = "Cannot resolve project root: " + ec.message();
        return false;
    }
    loose = std::filesystem::weakly_canonical(utf8_path(settings->globalize_path(path)), ec);
    if (ec) {
        error = "Cannot resolve settings_path: " + ec.message();
        return false;
    }
    auto rel = loose.lexically_relative(root);
    if (rel.empty() || rel == "." || *rel.begin() == ".." || rel.is_absolute()) {
        error = "settings_path escapes the project root.";
        return false;
    }
    return true;
}

bool read_document(const String& path, std::string& bytes, std::string& error) {
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    if (file.is_null()) {
        error = "Cannot read " + text(path) + " (Godot error " + std::to_string(FileAccess::get_open_error()) + ").";
        return false;
    }
    const PackedByteArray data = file->get_buffer(static_cast<int64_t>(file->get_length()));
    if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
        error = "Cannot finish reading " + text(path) + ".";
        return false;
    }
    bytes = data.is_empty() ? std::string()
                            : std::string(reinterpret_cast<const char*>(data.ptr()), static_cast<size_t>(data.size()));
    return true;
}

// Publish only a complete document, so a concurrent opener never parses a partially written file.
bool create_empty(const std::filesystem::path& loose, std::string& error) {
    std::error_code ec;
    std::filesystem::create_directories(loose.parent_path(), ec);
    if (ec) {
        error = "Cannot create settings directory: " + ec.message();
        return false;
    }
    static std::atomic<uint64_t> serial{0};
    auto temporary = loose;
    temporary += ".chorus-new-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                 std::to_string(serial++);
#if defined(_WIN32)
    int fd = _wopen(temporary.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    int fd = ::open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0666);
#endif
    if (fd < 0) {
        error = "Cannot exclusively create temporary settings document: " + std::string(std::strerror(errno));
        return false;
    }
    constexpr std::string_view empty = "{\"version\":1,\"generation\":{}}";
    std::string_view remaining = empty;
    bool written = true;
    while (!remaining.empty()) {
#if defined(_WIN32)
        const auto count = _write(fd, remaining.data(), static_cast<unsigned>(remaining.size()));
#else
        const auto count = ::write(fd, remaining.data(), remaining.size());
#endif
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            written = false;
            break;
        }
        remaining.remove_prefix(static_cast<size_t>(count));
    }
#if defined(_WIN32)
    const bool synced = _commit(fd) == 0;
    const bool closed = _close(fd) == 0;
#else
    const bool synced = ::fsync(fd) == 0;
    const bool closed = ::close(fd) == 0;
#endif
    if (!written || !synced || !closed) {
        std::filesystem::remove(temporary, ec);
        error = "Cannot finish temporary settings document.";
        if (ec)
            error += " Temporary cleanup failed: " + ec.message();
        return false;
    }
    std::filesystem::create_hard_link(temporary, loose, ec);
    std::error_code cleanup;
    std::filesystem::remove(temporary, cleanup);
    if (cleanup) {
        error = "Cannot remove temporary settings document: " + cleanup.message();
        return false;
    }
    if (ec && ec != std::errc::file_exists) {
        error = "Cannot publish empty settings document: " + ec.message();
        return false;
    }
    return true;
}

bool has_unrepresentable_text(const Chorus::ProviderOptionValue& value) {
    if (const auto* string = std::get_if<std::string>(&value))
        return string->find('\0') != std::string::npos;
    if (const auto* list = std::get_if<Chorus::ProviderOptionList>(&value)) {
        for (const auto& item : *list)
            if (has_unrepresentable_text(item))
                return true;
    }
    if (const auto* map = std::get_if<Chorus::ProviderOptionMap>(&value)) {
        for (const auto& [key, item] : *map)
            if (key.find('\0') != std::string::npos || has_unrepresentable_text(item))
                return true;
    }
    return false;
}

bool can_display(const Chorus::GenerationDefaults& value) {
    if (value.chat_template && value.chat_template->find('\0') != std::string::npos)
        return false;
    if (value.options.stop) {
        for (const auto& item : *value.options.stop)
            if (item.find('\0') != std::string::npos)
                return false;
    }
    if (value.options.constraint && std::holds_alternative<Chorus::OutputConstraint>(*value.options.constraint) &&
        std::get<Chorus::OutputConstraint>(*value.options.constraint).source.find('\0') != std::string::npos)
        return false;
    for (const auto& [key, options] : value.options.provider_options)
        if (key.find('\0') != std::string::npos || has_unrepresentable_text(options))
            return false;
    return true;
}

Variant option_variant(const Chorus::ProviderOptionValue& value) {
    if (const auto* p = std::get_if<bool>(&value))
        return *p;
    if (const auto* p = std::get_if<int64_t>(&value))
        return *p;
    if (const auto* p = std::get_if<double>(&value))
        return *p;
    if (const auto* p = std::get_if<std::string>(&value))
        return to_godot_string(*p);
    if (const auto* p = std::get_if<Chorus::ProviderOptionList>(&value)) {
        Array list;
        for (const auto& item : *p)
            list.push_back(option_variant(item));
        return list;
    }
    Dictionary map;
    for (const auto& [key, item] : std::get<Chorus::ProviderOptionMap>(value))
        map[to_godot_string(key)] = option_variant(item);
    return map;
}

Dictionary surface(const Chorus::GenerationDefaults& value) {
    Dictionary out;
    const auto& o = value.options;
#define PRESENT(NAME)                                                                                                  \
    if (o.NAME) {                                                                                                      \
        Dictionary selected;                                                                                           \
        selected["value"] = *o.NAME;                                                                                   \
        out[#NAME] = selected;                                                                                         \
    }
    PRESENT(max_tokens)
    PRESENT(temperature)
    PRESENT(top_k)
    PRESENT(top_p)
    PRESENT(frequency_penalty)
    PRESENT(presence_penalty)
    PRESENT(show_thinking)
#undef PRESENT
    if (o.seed) {
        Dictionary selected;
        selected["value"] = to_godot_string(std::to_string(*o.seed));
        out["seed"] = selected;
    }
    if (o.stop) {
        PackedStringArray stops;
        for (const auto& item : *o.stop)
            stops.push_back(to_godot_string(item));
        Dictionary selected;
        selected["value"] = stops;
        out["stop"] = selected;
    }
    if (o.constraint) {
        Dictionary constraint;
        if (std::holds_alternative<Chorus::UnconstrainedOutput>(*o.constraint))
            constraint["kind"] = "unconstrained";
        else {
            const auto& c = std::get<Chorus::OutputConstraint>(*o.constraint);
            switch (c.format) {
            case Chorus::ConstraintFormat::Gbnf:
                constraint["kind"] = "gbnf";
                break;
            case Chorus::ConstraintFormat::JsonSchema:
                constraint["kind"] = "json_schema";
                break;
            case Chorus::ConstraintFormat::Regex:
                constraint["kind"] = "regex";
                break;
            case Chorus::ConstraintFormat::Lark:
                constraint["kind"] = "lark";
                break;
            }
            constraint["source"] = to_godot_string(c.source);
        }
        Dictionary selected;
        selected["value"] = constraint;
        out["constraint"] = selected;
    }
    if (value.chat_template) {
        Dictionary selected;
        selected["value"] = to_godot_string(*value.chat_template);
        out["chat_template"] = selected;
    }
    Dictionary providers;
    for (const auto& [name, options] : o.provider_options)
        providers[to_godot_string(name)] = option_variant(options);
    out["provider_options"] = providers;
    return out;
}

void publish(const Chorus::GenerationDefaults& value) {
    ProjectSettings* settings = ProjectSettings::get_singleton();
    const Dictionary values = surface(value);
    for (const Dictionary& property : settings->get_property_list()) {
        const String name = property.get("name", String());
        if (name.begins_with(prefix) && name != selector && settings->has_setting(name))
            settings->clear(name);
    }
    for (const char* field : fields) {
        const String name = setting_name(field);
        settings->set_setting(name, values.get(field, Dictionary()));
        Dictionary info;
        info["name"] = name;
        info["type"] = Variant::DICTIONARY;
        settings->add_property_info(info);
        settings->set_initial_value(name, Dictionary());
    }
}

bool selected(const Variant& value, const char* name, Variant& result, bool& present, std::string& error) {
    if (value.get_type() != Variant::DICTIONARY) {
        error = std::string(name) + " must be a Dictionary.";
        return false;
    }
    const Dictionary wrapper = value;
    if (wrapper.is_empty()) {
        present = false;
        return true;
    }
    if (wrapper.size() != 1 || !wrapper.has("value")) {
        error = std::string(name) + " must be {} or contain only value.";
        return false;
    }
    present = true;
    result = wrapper["value"];
    return true;
}

bool convert_surface(Chorus::GenerationDefaults& result, std::string& error) {
    auto* settings = ProjectSettings::get_singleton();
    Chorus::GenerationDefaults next;
    Variant v;
    bool present;
#define GET(NAME)                                                                                                      \
    if (!selected(settings->get_setting(setting_name(#NAME)), #NAME, v, present, error))                               \
        return false;
#define INT32(NAME)                                                                                                    \
    GET(NAME)                                                                                                          \
    if (present) {                                                                                                     \
        if (v.get_type() != Variant::INT || (int64_t)v < INT32_MIN || (int64_t)v > INT32_MAX) {                        \
            error = #NAME " requires an int32 value.";                                                                 \
            return false;                                                                                              \
        }                                                                                                              \
        next.options.NAME = static_cast<int32_t>((int64_t)v);                                                          \
    }
#define FLOAT32(NAME)                                                                                                  \
    GET(NAME)                                                                                                          \
    if (present) {                                                                                                     \
        if (v.get_type() != Variant::FLOAT || !std::isfinite((double)v) ||                                             \
            std::abs((double)v) > std::numeric_limits<float>::max() ||                                                 \
            ((double)v != 0.0 && static_cast<float>((double)v) == 0.0f)) {                                             \
            error = #NAME " requires a representable finite float32 value.";                                           \
            return false;                                                                                              \
        }                                                                                                              \
        next.options.NAME = static_cast<float>((double)v);                                                             \
    }
    INT32(max_tokens)
    INT32(top_k)
    FLOAT32(temperature)
    FLOAT32(top_p)
    FLOAT32(frequency_penalty)
    FLOAT32(presence_penalty)
#undef INT32
#undef FLOAT32
    GET(seed)
    if (present) {
        if (v.get_type() != Variant::STRING) {
            error = "seed requires unsigned decimal text.";
            return false;
        }
        const std::string digits = text(v);
        uint64_t seed = 0;
        auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), seed);
        if (digits.empty() || ec != std::errc{} || end != digits.data() + digits.size()) {
            error = "seed requires unsigned uint64 decimal text.";
            return false;
        }
        next.options.seed = seed;
    }
    GET(show_thinking)
    if (present) {
        if (v.get_type() != Variant::BOOL) {
            error = "show_thinking requires bool.";
            return false;
        }
        next.options.show_thinking = (bool)v;
    }
    GET(stop)
    if (present) {
        std::vector<std::string> stops;
        if (v.get_type() == Variant::PACKED_STRING_ARRAY) {
            PackedStringArray array = v;
            for (const String& item : array)
                stops.push_back(text(item));
        } else if (v.get_type() == Variant::ARRAY) {
            Array array = v;
            for (int i = 0; i < array.size(); ++i) {
                if (array[i].get_type() != Variant::STRING) {
                    error = "stop requires an array of strings.";
                    return false;
                }
                stops.push_back(text(array[i]));
            }
        } else {
            error = "stop requires an array of strings.";
            return false;
        }
        next.options.stop = std::move(stops);
    }
    GET(chat_template)
    if (present) {
        if (v.get_type() != Variant::STRING) {
            error = "chat_template requires string.";
            return false;
        }
        next.chat_template = text(v);
    }
    GET(constraint)
    if (present) {
        if (v.get_type() != Variant::DICTIONARY) {
            error = "constraint requires a Dictionary.";
            return false;
        }
        Dictionary c = v;
        if (!c.has("kind") || c["kind"].get_type() != Variant::STRING) {
            error = "constraint.kind requires string.";
            return false;
        }
        String kind = c["kind"];
        if (kind == "unconstrained" && c.size() == 1)
            next.options.constraint = Chorus::UnconstrainedOutput{};
        else {
            Chorus::ConstraintFormat format;
            if (kind == "gbnf")
                format = Chorus::ConstraintFormat::Gbnf;
            else if (kind == "json_schema")
                format = Chorus::ConstraintFormat::JsonSchema;
            else if (kind == "regex")
                format = Chorus::ConstraintFormat::Regex;
            else if (kind == "lark")
                format = Chorus::ConstraintFormat::Lark;
            else {
                error = "constraint.kind is invalid.";
                return false;
            }
            if (c.size() != 2 || !c.has("source") || c["source"].get_type() != Variant::STRING) {
                error = "constraint requires exactly kind and string source.";
                return false;
            }
            next.options.constraint = Chorus::OutputConstraint{format, text(c["source"])};
        }
    }
#undef GET
    const Variant provider_value = settings->get_setting(setting_name("provider_options"));
    if (provider_value.get_type() != Variant::DICTIONARY) {
        error = "provider_options must be a Dictionary.";
        return false;
    }
    const Dictionary providers = provider_value;
    for (const Variant& provider_key : providers.keys()) {
        if (provider_key.get_type() != Variant::STRING) {
            error = "provider_options contains a non-string namespace key.";
            return false;
        }
        const std::string name = text(provider_key);
        const Variant& namespace_value = providers[provider_key];
        if (namespace_value.get_type() != Variant::DICTIONARY) {
            error = "provider_options." + name + " must be a Dictionary.";
            return false;
        }
        Chorus::ProviderOptionMap options;
        const Dictionary names = namespace_value;
        for (const Variant& key : names.keys()) {
            if (key.get_type() != Variant::STRING) {
                error = "provider_options." + name + " contains a non-string option key.";
                return false;
            }
            auto item = variant_to_option_value(names[key], error);
            if (!item) {
                error = "provider_options." + name + "." + text(key) + " " + error;
                return false;
            }
            options.emplace(text(key), std::move(*item));
        }
        if (!options.empty())
            next.options.provider_options.emplace(name, std::move(options));
    }
    const auto checked = chorus_host_settings::serialize_generation_defaults(next);
    if (!checked.ok()) {
        error = checked.path + ": " + checked.error;
        return false;
    }
    result = std::move(next);
    return true;
}

bool import_selected(const String& path, std::string& error) {
    std::filesystem::path loose;
    if (!contained_path(path, loose, error))
        return false;
    std::string bytes;
    if (!FileAccess::file_exists(path)) {
        std::error_code inspect;
        const bool existing = std::filesystem::exists(loose, inspect);
        if (inspect) {
            error = "Cannot inspect settings file: " + inspect.message();
            return false;
        }
        if (existing) {
            error = "Selected existing settings file is unreadable.";
            return false;
        }
        if (loose.empty()) {
            error = "Cannot create a missing settings file in a packed project.";
            return false;
        }
        if (!create_empty(loose, error))
            return false;
    }
    if (!read_document(path, bytes, error))
        return false;
    auto parsed = chorus_host_settings::parse_generation_defaults(bytes);
    if (!parsed.ok()) {
        error = text(path) + ": " + parsed.path + ": " + parsed.error;
        return false;
    }
    if (!can_display(parsed.defaults)) {
        error = text(path) + ": embedded NUL cannot be represented by Godot String; file was not imported.";
        return false;
    }
    publish(parsed.defaults);
    auto& current = state.value();
    current.value = std::move(parsed.defaults);
    current.path = path;
    current.failed = false;
    current.imported_bytes = std::move(bytes);
    return true;
}

} // namespace

void initialize_project_generation_defaults() {
    state.emplace();
    auto* settings = ProjectSettings::get_singleton();
    if (!settings->has_setting(selector))
        settings->set_setting(selector, "res://chorus/settings.json");
    Dictionary info;
    info["name"] = selector;
    info["type"] = Variant::STRING;
    info["hint"] = PROPERTY_HINT_GLOBAL_FILE;
    info["hint_string"] = "*.json";
    settings->add_property_info(info);
    settings->set_initial_value(selector, "res://chorus/settings.json");
    std::string error;
    const Variant chosen = settings->get_setting(selector);
    state->path = chosen.get_type() == Variant::STRING ? String(chosen) : String("res://chorus/settings.json");
    if (chosen.get_type() != Variant::STRING || !import_selected(chosen, error)) {
        if (error.empty())
            error = "settings_path must be a res:// string.";
        publish(state->value);
        UtilityFunctions::push_error("[Chorus] generation settings import failed: " + to_godot_string(error));
    }
}

void shutdown_project_generation_defaults() {
    state.reset();
}

bool reload_project_generation_defaults(std::string& error) {
    if (!state) {
        error = "Project defaults are not initialized.";
        return false;
    }
    const Variant selected_path = ProjectSettings::get_singleton()->get_setting(selector);
    if (selected_path.get_type() != Variant::STRING)
        error = "settings_path must be a res:// string.";
    else if (import_selected(selected_path, error))
        return true;
    ProjectSettings::get_singleton()->set_setting(
        selector, state->path.is_empty() ? String("res://chorus/settings.json") : state->path
    );
    UtilityFunctions::push_error("[Chorus] generation settings reload failed: " + to_godot_string(error));
    return false;
}

SettingsResult sync_project_generation_defaults() {
    Chorus::GenerationDefaults choices;
    std::string error;
    if (!project_generation_defaults(choices, error))
        return {SettingsStatus::Invalid, error};
    return {SettingsStatus::Ok, {}};
}

SettingsResult save_project_generation_defaults() {
    if (!state)
        return {SettingsStatus::SourceFailed, "Project defaults are not initialized."};
    auto synced = sync_project_generation_defaults();
    if (synced.status != SettingsStatus::Ok)
        return synced;
    if (state->failed)
        return {SettingsStatus::SourceFailed, "The settings source failed. Reload generation defaults before saving."};

    std::filesystem::path loose;
    std::string error;
    if (!contained_path(state->path, loose, error))
        return {SettingsStatus::Invalid, error};
    if (loose.empty())
        return {SettingsStatus::IoError, "Cannot save a packed settings resource without a writable project root."};
    std::string current;
    if (!read_document(state->path, current, error))
        return {SettingsStatus::IoError, error};
    if (current != state->imported_bytes)
        return {SettingsStatus::Conflict, "Generation settings changed outside the editor. Reload before saving."};
    const auto parsed = chorus_host_settings::parse_generation_defaults(current);
    if (!parsed.ok())
        return {SettingsStatus::Invalid, "Destination is malformed: " + parsed.path + ": " + parsed.error};
    const auto serialized = chorus_host_settings::serialize_generation_defaults(state->value);
    if (!serialized.ok())
        return {SettingsStatus::Invalid, serialized.path + ": " + serialized.error};

    static std::atomic<uint64_t> serial{0};
    auto temporary = loose;
    temporary += ".chorus-save-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                 std::to_string(serial++);
#if defined(_WIN32)
    const int fd = _wopen(temporary.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    const int fd = ::open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0666);
#endif
    if (fd < 0)
        return {SettingsStatus::IoError, "Cannot create temporary settings file: " + std::string(std::strerror(errno))};
    std::string_view remaining = serialized.json;
    bool written = true;
    while (!remaining.empty()) {
#if defined(_WIN32)
        const auto count = _write(fd, remaining.data(), static_cast<unsigned>(remaining.size()));
#else
        const auto count = ::write(fd, remaining.data(), remaining.size());
#endif
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            written = false;
            break;
        }
        remaining.remove_prefix(static_cast<size_t>(count));
    }
#if defined(_WIN32)
    const bool synced_file = _commit(fd) == 0;
    const bool closed = _close(fd) == 0;
#else
    const bool synced_file = ::fsync(fd) == 0;
    const bool closed = ::close(fd) == 0;
#endif
    if (!written || !synced_file || !closed) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return {SettingsStatus::IoError, "Cannot finish temporary settings file."};
    }
    std::string rechecked;
    if (!read_document(state->path, rechecked, error) || rechecked != current) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return {
            SettingsStatus::Conflict,
            "Generation settings changed or became unreadable before replacement. Reload before saving."
        };
    }
    std::error_code ec;
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), loose.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
#else
    std::filesystem::rename(temporary, loose, ec);
#endif
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return {SettingsStatus::IoError, "Cannot replace settings file: " + ec.message()};
    }
    state->imported_bytes = serialized.json;
    return {SettingsStatus::Ok, {}};
}

bool project_generation_defaults(Chorus::GenerationDefaults& out, std::string& error) {
    if (!state) {
        error = "Project defaults are not initialized.";
        return false;
    }
    auto* settings = ProjectSettings::get_singleton();
    const Variant choice = settings->get_setting(selector);
    if (choice.get_type() != Variant::STRING) {
        settings->set_setting(selector, state->path);
        UtilityFunctions::push_error("[Chorus] settings_path must be a res:// string.");
    } else if (String(choice) != state->path) {
        if (!reload_project_generation_defaults(error))
            error.clear();
    }
    if (!convert_surface(out, error)) {
        error = "chorus/generation/" + error;
        return false;
    }
    state->value = out;
    return true;
}

} // namespace godot_chorus
