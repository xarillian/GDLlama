#include "host_settings/generation_defaults_codec.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chorus_host_settings {
namespace {
using Json = nlohmann::json;

struct CodecError : std::runtime_error {
    std::string path;
    CodecError(std::string at, std::string why) : std::runtime_error(std::move(why)), path(std::move(at)) {}
};

std::string child(const std::string& path, const std::string& key) {
    std::string result = path;
    result += '/';
    for (char c : key) {
        if (c == '~')
            result += "~0";
        else if (c == '/')
            result += "~1";
        else
            result += c;
    }
    return result;
}

void require(bool valid, const std::string& path, const char* reason) {
    if (!valid)
        throw CodecError(path, reason);
}

void valid_text(const std::string& text, const std::string& path) {
    try {
        (void)Json(text).dump(-1, ' ', false, Json::error_handler_t::strict);
    } catch (const Json::exception& error) {
        throw CodecError(path, error.what());
    }
}

void keys(const Json& object, const std::string& path, std::initializer_list<const char*> allowed) {
    require(object.is_object(), path, "expected object");
    for (auto it = object.begin(); it != object.end(); ++it) {
        bool known = false;
        for (auto key : allowed)
            known |= it.key() == key;
        require(known, child(path, it.key()), "unknown field");
    }
}

int64_t integer(const Json& value, const std::string& path) {
    require(value.is_number_integer(), path, "expected signed integer literal");
    if (value.is_number_unsigned()) {
        auto n = value.get<uint64_t>();
        require(n <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), path, "integer out of range");
        return static_cast<int64_t>(n);
    }
    return value.get<int64_t>();
}

uint64_t unsigned_integer(const Json& value, const std::string& path) {
    require(value.is_number_integer(), path, "expected nonnegative integer literal");
    if (value.is_number_unsigned())
        return value.get<uint64_t>();
    auto n = value.get<int64_t>();
    require(n >= 0, path, "expected nonnegative integer");
    return static_cast<uint64_t>(n);
}

int32_t int32(const Json& value, const std::string& path) {
    const auto n = integer(value, path);
    require(n >= INT32_MIN && n <= INT32_MAX, path, "integer out of range");
    return static_cast<int32_t>(n);
}

float float32(const Json& value, const std::string& path) {
    require(value.is_number(), path, "expected number");
    const auto n = value.get<double>();
    const auto f = static_cast<float>(n);
    require(std::isfinite(n) && std::isfinite(f) && (n == 0 || f != 0), path, "float32 out of range");
    return f;
}

Chorus::ProviderOptionValue option(const Json& value, const std::string& path, size_t depth) {
    using namespace Chorus;
    require(!value.is_null(), path, "null is not a value");
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_number_integer())
        return integer(value, path);
    if (value.is_number_float()) {
        const double number = value.get<double>();
        require(std::isfinite(number), path, "nonfinite number");
        return number;
    }
    if (value.is_string())
        return value.get<std::string>();
    require(value.is_array() || value.is_object(), path, "unsupported option type");
    require(depth < 64, path, "option container nesting exceeds 64");
    if (value.is_array()) {
        ProviderOptionList list;
        for (size_t i = 0; i < value.size(); ++i)
            list.push_back(option(value[i], child(path, std::to_string(i)), depth + 1));
        return list;
    }
    ProviderOptionMap map;
    for (auto it = value.begin(); it != value.end(); ++it)
        map.emplace(it.key(), option(it.value(), child(path, it.key()), depth + 1));
    return map;
}

Json option_json(const Chorus::ProviderOptionValue& value, const std::string& path, size_t depth) {
    using namespace Chorus;
    if (const auto* v = std::get_if<bool>(&value))
        return *v;
    if (const auto* v = std::get_if<int64_t>(&value))
        return *v;
    if (const auto* v = std::get_if<double>(&value)) {
        require(std::isfinite(*v), path, "nonfinite number");
        return *v;
    }
    if (const auto* v = std::get_if<std::string>(&value)) {
        valid_text(*v, path);
        return *v;
    }
    require(depth < 64, path, "option container nesting exceeds 64");
    if (const auto* v = std::get_if<ProviderOptionList>(&value)) {
        Json result = Json::array();
        for (size_t i = 0; i < v->size(); ++i)
            result.push_back(option_json((*v)[i], child(path, std::to_string(i)), depth + 1));
        return result;
    }
    Json result = Json::object();
    const auto& map = std::get<ProviderOptionMap>(value);
    for (const auto& [key, item] : map) {
        valid_text(key, child(path, key));
        result[key] = option_json(item, child(path, key), depth + 1);
    }
    return result;
}

Chorus::ConstraintChoice constraint(const Json& value, const std::string& path) {
    keys(value, path, {"kind", "source"});
    require(value.contains("kind") && value["kind"].is_string(), child(path, "kind"), "expected kind string");
    const auto kind = value["kind"].get<std::string>();
    if (kind == "unconstrained") {
        require(value.size() == 1, path, "unconstrained must not have a source");
        return Chorus::UnconstrainedOutput{};
    }
    Chorus::ConstraintFormat format;
    if (kind == "gbnf")
        format = Chorus::ConstraintFormat::Gbnf;
    else if (kind == "json_schema")
        format = Chorus::ConstraintFormat::JsonSchema;
    else if (kind == "regex")
        format = Chorus::ConstraintFormat::Regex;
    else if (kind == "lark")
        format = Chorus::ConstraintFormat::Lark;
    else
        throw CodecError(child(path, "kind"), "unknown constraint kind");
    require(value.contains("source") && value["source"].is_string(), child(path, "source"), "expected source string");
    return Chorus::OutputConstraint{format, value["source"].get<std::string>()};
}

struct FloatingTokenCheck : nlohmann::json_sax<Json> {
    struct Frame {
        std::string path;
        std::string key;
        size_t index = 0;
        bool array = false;
    };
    std::vector<Frame> frames;
    std::string path;
    std::string error;

    std::string location() const {
        if (frames.empty())
            return "/";
        const auto& parent = frames.back();
        return child(parent.path, parent.array ? std::to_string(parent.index) : parent.key);
    }
    bool scalar() {
        if (!frames.empty() && frames.back().array)
            ++frames.back().index;
        return true;
    }
    bool null() override { return scalar(); }
    bool boolean(bool) override { return scalar(); }
    bool number_integer(number_integer_t) override { return scalar(); }
    bool number_unsigned(number_unsigned_t) override { return scalar(); }
    bool number_float(number_float_t value, const string_t& token) override {
        if (token.find_first_of(".eE") == std::string::npos) {
            path = location();
            error = "integer literal out of range";
            return false;
        }
        const auto mantissa_end = std::find_if(token.begin(), token.end(), [](char c) { return c == 'e' || c == 'E'; });
        if (value == 0 && std::any_of(token.begin(), mantissa_end, [](char c) { return c >= '1' && c <= '9'; })) {
            path = location();
            error = "floating literal underflows double";
            return false;
        }
        return scalar();
    }
    bool string(string_t&) override { return scalar(); }
    bool binary(binary_t&) override { return scalar(); }
    bool start_object(size_t) override {
        frames.push_back({location(), {}, 0, false});
        return true;
    }
    bool start_array(size_t) override {
        frames.push_back({location(), {}, 0, true});
        return true;
    }
    bool key(string_t& key) override {
        frames.back().key = key;
        return true;
    }
    bool end_object() override { return finish(); }
    bool end_array() override { return finish(); }
    bool finish() {
        frames.pop_back();
        return scalar();
    }
    bool parse_error(size_t, const std::string&, const nlohmann::detail::exception& exception) override {
        path = location();
        error = exception.what();
        return false;
    }
};

const char* kind(Chorus::ConstraintFormat format) {
    switch (format) {
    case Chorus::ConstraintFormat::Gbnf:
        return "gbnf";
    case Chorus::ConstraintFormat::JsonSchema:
        return "json_schema";
    case Chorus::ConstraintFormat::Regex:
        return "regex";
    case Chorus::ConstraintFormat::Lark:
        return "lark";
    }
    throw CodecError("/generation/constraint/kind", "unknown constraint kind");
}

} // namespace

ParseResult parse_generation_defaults(std::string_view bytes) {
    try {
        const auto nul = bytes.find('\0');
        if (nul != std::string_view::npos)
            throw CodecError("/", "raw NUL byte at offset " + std::to_string(nul) + " is not valid JSON");
        FloatingTokenCheck numeric_check;
        if (!Json::sax_parse(bytes.begin(), bytes.end(), &numeric_check))
            throw CodecError(numeric_check.path, numeric_check.error);
        struct Frame {
            std::string path;
            std::set<std::string> keys;
            std::string current_key;
            size_t index = 0;
            bool array = false;
        };
        std::vector<Frame> frames;
        auto callback = [&frames](int, Json::parse_event_t event, Json& parsed) {
            using Event = Json::parse_event_t;
            if (event == Event::object_start || event == Event::array_start) {
                std::string path;
                if (!frames.empty()) {
                    auto& parent = frames.back();
                    path = child(parent.path, parent.array ? std::to_string(parent.index) : parent.current_key);
                }
                frames.push_back({std::move(path), {}, {}, 0, event == Event::array_start});
            } else if (event == Event::key) {
                auto& parent = frames.back();
                const auto& key = parsed.get_ref<const std::string&>();
                if (!parent.keys.insert(key).second)
                    throw CodecError(child(parent.path, key), "duplicate key");
                parent.current_key = key;
            } else if (event == Event::object_end || event == Event::array_end) {
                frames.pop_back();
                if (!frames.empty() && frames.back().array)
                    ++frames.back().index;
            } else if (event == Event::value && !frames.empty() && frames.back().array) {
                ++frames.back().index;
            }
            return true;
        };
        const Json document = Json::parse(bytes.begin(), bytes.end(), callback);
        keys(document, "", {"version", "generation"});
        require(
            document.contains("version") && document["version"].is_number_integer() &&
                unsigned_integer(document["version"], "/version") == 1,
            "/version",
            "unsupported version"
        );
        require(document.contains("generation"), "/generation", "missing generation");
        const Json& g = document["generation"];
        keys(
            g,
            "/generation",
            {"max_tokens",
             "temperature",
             "top_k",
             "top_p",
             "seed",
             "frequency_penalty",
             "presence_penalty",
             "stop",
             "show_thinking",
             "constraint",
             "provider_options",
             "chat_template"}
        );
        Chorus::GenerationDefaults result;
        auto& options = result.options;
        if (g.contains("max_tokens"))
            options.max_tokens = int32(g["max_tokens"], "/generation/max_tokens");
        if (g.contains("temperature"))
            options.temperature = float32(g["temperature"], "/generation/temperature");
        if (g.contains("top_k"))
            options.top_k = int32(g["top_k"], "/generation/top_k");
        if (g.contains("top_p"))
            options.top_p = float32(g["top_p"], "/generation/top_p");
        if (g.contains("seed"))
            options.seed = unsigned_integer(g["seed"], "/generation/seed");
        if (g.contains("frequency_penalty"))
            options.frequency_penalty = float32(g["frequency_penalty"], "/generation/frequency_penalty");
        if (g.contains("presence_penalty"))
            options.presence_penalty = float32(g["presence_penalty"], "/generation/presence_penalty");
        if (g.contains("show_thinking")) {
            require(g["show_thinking"].is_boolean(), "/generation/show_thinking", "expected boolean");
            options.show_thinking = g["show_thinking"].get<bool>();
        }
        if (g.contains("stop")) {
            require(g["stop"].is_array(), "/generation/stop", "expected array");
            std::vector<std::string> stop;
            for (size_t i = 0; i < g["stop"].size(); ++i) {
                const auto path = child("/generation/stop", std::to_string(i));
                require(g["stop"][i].is_string(), path, "expected string");
                stop.push_back(g["stop"][i].get<std::string>());
            }
            options.stop = std::move(stop);
        }
        if (g.contains("constraint"))
            options.constraint = constraint(g["constraint"], "/generation/constraint");
        if (g.contains("chat_template")) {
            require(g["chat_template"].is_string(), "/generation/chat_template", "expected string");
            result.chat_template = g["chat_template"].get<std::string>();
        }
        if (g.contains("provider_options")) {
            const auto& providers = g["provider_options"];
            require(providers.is_object(), "/generation/provider_options", "expected object");
            for (auto it = providers.begin(); it != providers.end(); ++it) {
                const auto path = child("/generation/provider_options", it.key());
                require(it.value().is_object(), path, "expected namespace object");
                Chorus::ProviderOptionMap choices;
                for (auto entry = it.value().begin(); entry != it.value().end(); ++entry)
                    choices.emplace(entry.key(), option(entry.value(), child(path, entry.key()), 0));
                if (!choices.empty())
                    options.provider_options.emplace(it.key(), Chorus::ProviderOptionValue{std::move(choices)});
            }
        }
        return {std::move(result), {}, {}};
    } catch (const CodecError& error) {
        return {{}, error.path, error.what()};
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        return {{}, "/", error.what()};
    }
}

SerializeResult serialize_generation_defaults(const Chorus::GenerationDefaults& defaults) {
    std::string path = "/generation";
    try {
        const auto& o = defaults.options;
        Json g = Json::object();
        if (o.max_tokens)
            g["max_tokens"] = *o.max_tokens;
        if (o.temperature) {
            path = "/generation/temperature";
            require(std::isfinite(*o.temperature), path, "nonfinite number");
            g["temperature"] = *o.temperature;
        }
        if (o.top_k)
            g["top_k"] = *o.top_k;
        if (o.top_p) {
            path = "/generation/top_p";
            require(std::isfinite(*o.top_p), path, "nonfinite number");
            g["top_p"] = *o.top_p;
        }
        if (o.seed)
            g["seed"] = *o.seed;
        if (o.frequency_penalty) {
            path = "/generation/frequency_penalty";
            require(std::isfinite(*o.frequency_penalty), path, "nonfinite number");
            g["frequency_penalty"] = *o.frequency_penalty;
        }
        if (o.presence_penalty) {
            path = "/generation/presence_penalty";
            require(std::isfinite(*o.presence_penalty), path, "nonfinite number");
            g["presence_penalty"] = *o.presence_penalty;
        }
        if (o.stop) {
            for (size_t i = 0; i < o.stop->size(); ++i)
                valid_text((*o.stop)[i], child("/generation/stop", std::to_string(i)));
            g["stop"] = *o.stop;
        }
        if (o.show_thinking)
            g["show_thinking"] = *o.show_thinking;
        if (o.constraint) {
            if (std::holds_alternative<Chorus::UnconstrainedOutput>(*o.constraint))
                g["constraint"] = {{"kind", "unconstrained"}};
            else {
                const auto& value = std::get<Chorus::OutputConstraint>(*o.constraint);
                valid_text(value.source, "/generation/constraint/source");
                g["constraint"] = {{"kind", kind(value.format)}, {"source", value.source}};
            }
        }
        if (defaults.chat_template) {
            valid_text(*defaults.chat_template, "/generation/chat_template");
            g["chat_template"] = *defaults.chat_template;
        }
        if (!o.provider_options.empty()) {
            Json providers = Json::object();
            for (const auto& [name, value] : o.provider_options) {
                path = child("/generation/provider_options", name);
                valid_text(name, path);
                const auto* map = std::get_if<Chorus::ProviderOptionMap>(&value);
                require(map != nullptr, path, "expected namespace object");
                if (map->empty())
                    continue;
                Json choices = Json::object();
                for (const auto& [key, item] : *map) {
                    valid_text(key, child(path, key));
                    choices[key] = option_json(item, child(path, key), 0);
                }
                providers[name] = std::move(choices);
            }
            if (!providers.empty())
                g["provider_options"] = std::move(providers);
        }
        path = "/generation";
        return {"{\"version\":1,\"generation\":" + g.dump(-1, ' ', false, Json::error_handler_t::strict) + "}", {}, {}};
    } catch (const CodecError& error) {
        return {{}, error.path, error.what()};
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        return {{}, path, error.what()};
    }
}

} // namespace chorus_host_settings
