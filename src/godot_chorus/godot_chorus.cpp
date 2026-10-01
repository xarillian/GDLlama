#include "godot_chorus/godot_chorus.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <variant>

#include "chorus/engine_factory.hpp"
#include "godot_chorus/option_conversion.hpp"
#include "godot_chorus/project_generation_defaults.hpp"
#include "godot_chorus/provider_option_properties.hpp"
#include "godot_chorus/request_conversion.hpp"

#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

static const char* chorus_error_name(Chorus::ChorusError e) {
    switch (e) {
    case Chorus::ChorusError::None:
        return "None";
    case Chorus::ChorusError::ModelLoad:
        return "ModelLoad";
    case Chorus::ChorusError::ContextInit:
        return "ContextInit";
    case Chorus::ChorusError::Decode:
        return "Decode";
    case Chorus::ChorusError::Tokenize:
        return "Tokenize";
    case Chorus::ChorusError::InvalidRequest:
        return "InvalidRequest";
    case Chorus::ChorusError::EngineNotReady:
        return "EngineNotReady";
    case Chorus::ChorusError::Cancelled:
        return "Cancelled";
    case Chorus::ChorusError::UnsupportedModelFormat:
        return "UnsupportedModelFormat";
    case Chorus::ChorusError::UnsupportedFeature:
        return "UnsupportedFeature";
    case Chorus::ChorusError::UnsupportedOption:
        return "UnsupportedOption";
    case Chorus::ChorusError::SessionBusy:
        return "SessionBusy";
    case Chorus::ChorusError::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

static ChorusLoadPhase::Value to_godot_load_phase(Chorus::LoadPhase phase) {
    switch (phase) {
    case Chorus::LoadPhase::ReleasingEngine:
        return ChorusLoadPhase::RELEASING_ENGINE;
    case Chorus::LoadPhase::LoadingModel:
        return ChorusLoadPhase::LOADING_MODEL;
    case Chorus::LoadPhase::InitializingEngine:
        return ChorusLoadPhase::INITIALIZING_ENGINE;
    }
    return ChorusLoadPhase::LOADING_MODEL;
}

static Chorus::Provider to_chorus_provider(GodotChorus::ProviderChoice provider) {
    return provider == GodotChorus::PROVIDER_ECHO ? Chorus::Provider::Echo : Chorus::Provider::Llama;
}

using godot_chorus::to_godot_string;

static Ref<ChorusSubmitResult>
submit_result(const Ref<ChorusInferenceRequest>& request, const Chorus::SubmitResult& result) {
    return Ref<ChorusSubmitResult>(memnew(ChorusSubmitResult(
        request,
        result.request_id,
        result.request_message_id.value_or(-1),
        result.response_message_id.value_or(-1),
        GodotChorus::to_godot(result.error),
        to_godot_string(result.message)
    )));
}

static Ref<ChorusSubmitResult> rejected_submit(const Ref<ChorusInferenceRequest>& request, const String& message) {
    return Ref<ChorusSubmitResult>(
        memnew(ChorusSubmitResult(request, -1, -1, -1, GodotChorus::ERR_INVALID_REQUEST, message))
    );
}

static Ref<ChorusResult> operation_result(std::optional<Chorus::ChorusError> error) {
    return Ref<ChorusResult>(memnew(ChorusResult(
        error ? GodotChorus::to_godot(*error) : GodotChorus::ERR_NONE,
        error ? String(chorus_error_name(*error)) : String()
    )));
}

int GodotChorus::to_godot(Chorus::TurnOutcome outcome) {
    switch (outcome) {
    case Chorus::TurnOutcome::None:
        return TURN_NONE;
    case Chorus::TurnOutcome::Completed:
        return TURN_COMPLETED;
    case Chorus::TurnOutcome::Cancelled:
        return TURN_CANCELLED;
    case Chorus::TurnOutcome::Errored:
        return TURN_ERRORED;
    }
    return TURN_NONE;
}

int GodotChorus::to_godot(Chorus::LogLevel level) {
    switch (level) {
    case Chorus::LogLevel::Debug:
        return LOG_DEBUG;
    case Chorus::LogLevel::Info:
        return LOG_INFO;
    case Chorus::LogLevel::Warn:
        return LOG_WARN;
    case Chorus::LogLevel::Error:
        return LOG_ERROR;
    case Chorus::LogLevel::Fatal:
        return LOG_FATAL;
    case Chorus::LogLevel::Off:
        return LOG_OFF;
    }
    return LOG_INFO;
}

int GodotChorus::to_godot(Chorus::ChorusError e) {
    switch (e) {
    case Chorus::ChorusError::None:
        return ERR_NONE;
    case Chorus::ChorusError::ModelLoad:
        return ERR_MODEL_LOAD;
    case Chorus::ChorusError::ContextInit:
        return ERR_CONTEXT_INIT;
    case Chorus::ChorusError::Decode:
        return ERR_DECODE;
    case Chorus::ChorusError::Tokenize:
        return ERR_TOKENIZE;
    case Chorus::ChorusError::InvalidRequest:
        return ERR_INVALID_REQUEST;
    case Chorus::ChorusError::EngineNotReady:
        return ERR_ENGINE_NOT_READY;
    case Chorus::ChorusError::Cancelled:
        return ERR_CANCELLED;
    case Chorus::ChorusError::UnsupportedModelFormat:
        return ERR_UNSUPPORTED_MODEL_FORMAT;
    case Chorus::ChorusError::UnsupportedFeature:
        return ERR_UNSUPPORTED_FEATURE;
    case Chorus::ChorusError::UnsupportedOption:
        return ERR_UNSUPPORTED_OPTION;
    case Chorus::ChorusError::SessionBusy:
        return ERR_SESSION_BUSY;
    case Chorus::ChorusError::Unknown:
        return ERR_UNKNOWN;
    }
    return ERR_UNKNOWN;
}

// Lifecycle

void GodotChorus::_notification(int p_what) {
    if (p_what == NOTIFICATION_READY) {
        set_process(true);
    } else if (p_what == NOTIFICATION_EXIT_TREE) {
        // `GodotChorus::_process` stops with the tree, so the last frame's
        // diagnostics would otherwise remain buffered when the node is freed.
        // The node is fully alive here, making signal emission safe.
        drain_logs();
    }
}

// Keeping `LOG_LEVEL_SETTING` here puts its declaration beside every read.
static const char* LOG_LEVEL_SETTING = "chorus/logging/min_level";

// Godot-visible values in `GodotChorus::LogLevelCode` order.
static const char* LOG_LEVEL_HINT = "Debug,Info,Warning,Error,Fatal,Off";

void GodotChorus::register_project_settings() {
    ProjectSettings* settings = ProjectSettings::get_singleton();
    if (settings == nullptr)
        return;

    // `godot::OS::has_feature("editor")` separates editor or debug runs from
    // exported games, which is the line between the two defaults.
    const bool in_editor = OS::get_singleton() != nullptr && OS::get_singleton()->has_feature("editor");
    const int64_t default_level = in_editor ? LOG_INFO : (int64_t)Chorus::log_level_default;

    if (!settings->has_setting(LOG_LEVEL_SETTING))
        settings->set_setting(LOG_LEVEL_SETTING, default_level);

    // Declared every launch, not only when the setting is first created: the
    // hint and initial value live in memory, and a project that saved a custom
    // value would otherwise lose the dropdown on its next editor start.
    Dictionary info;
    info["name"] = LOG_LEVEL_SETTING;
    info["type"] = Variant::INT;
    info["hint"] = PROPERTY_HINT_ENUM;
    info["hint_string"] = LOG_LEVEL_HINT;
    settings->add_property_info(info);
    settings->set_initial_value(LOG_LEVEL_SETTING, default_level);
}

Chorus::LogLevel GodotChorus::effective_log_level() const {
    int64_t level = _override_log_level ? _log_level : (int64_t)Chorus::log_level_default;

    if (!_override_log_level) {
        ProjectSettings* settings = ProjectSettings::get_singleton();
        if (settings != nullptr && settings->has_setting(LOG_LEVEL_SETTING))
            level = (int64_t)settings->get_setting(LOG_LEVEL_SETTING);
    }

    // A project file is editable by hand, so an out-of-range value reaches
    // here; the quiet default beats undefined behavior on the enum.
    if (level < LOG_DEBUG || level > LOG_OFF)
        return Chorus::log_level_default;
    return (Chorus::LogLevel)level;
}

void GodotChorus::drain_logs() {
    for (const auto& record : _runtime.poll_logs()) {
        Dictionary fields;
        for (const auto& field : record.fields) {
            const String key = to_godot_string(field.first);
            if (const auto* integer = std::get_if<int64_t>(&field.second))
                fields[key] = *integer;
            else if (const auto* number = std::get_if<double>(&field.second))
                fields[key] = *number;
            else if (const auto* flag = std::get_if<bool>(&field.second))
                fields[key] = *flag;
            else
                fields[key] = to_godot_string(std::get<std::string>(field.second));
        }

        // Unix seconds, accepted by `Time.get_datetime_string_from_unix_time`
        // and related Godot APIs. Production stamps the record, so this time
        // predates the frame that delivers it.
        const double produced_at = std::chrono::duration<double>(record.timestamp.time_since_epoch()).count();

        emit_signal(
            "log_record",
            to_godot(record.level),
            to_godot_string(record.message),
            fields,
            record.request_id.value_or(-1),
            record.session_id ? StringName(to_godot_string(*record.session_id)) : StringName(),
            produced_at
        );
    }
}

void GodotChorus::_process(double /*delta*/) {
    drain_logs();
    for (const auto& event : _runtime.poll()) {
        const StringName session = event.session_id ? StringName(to_godot_string(*event.session_id)) : StringName();
        switch (event.kind) {
        case Chorus::RuntimeEvent::Kind::StreamedToken:
            emit_signal("token_generated", event.request_id, session, to_godot_string(event.text));
            break;
        case Chorus::RuntimeEvent::Kind::StreamedReasoningToken:
            emit_signal("reasoning_token_generated", event.request_id, session, to_godot_string(event.text));
            break;
        case Chorus::RuntimeEvent::Kind::HistoryTruncated: {
            PackedInt64Array omitted;
            omitted.resize(static_cast<int64_t>(event.omitted_message_ids.size()));
            for (int64_t i = 0; i < omitted.size(); ++i)
                omitted.set(i, event.omitted_message_ids[static_cast<size_t>(i)]);
            emit_signal("history_truncated", event.request_id, session, omitted);
            break;
        }
        case Chorus::RuntimeEvent::Kind::PromptRendered: {
            PackedInt64Array omitted;
            for (auto id : event.omitted_message_ids)
                omitted.push_back(id);
            emit_signal("prompt_rendered", event.request_id, session, to_godot_string(event.text), omitted);
            break;
        }
        case Chorus::RuntimeEvent::Kind::MessageTokenCount:
            emit_signal("message_token_counted", event.request_id, event.token_count);
            break;
        case Chorus::RuntimeEvent::Kind::Complete:
            emit_signal(
                "generation_complete",
                event.request_id,
                session,
                event.message_id.value_or(-1),
                to_godot_string(event.text),
                to_godot_string(event.reasoning)
            );
            break;
        case Chorus::RuntimeEvent::Kind::Embedding: {
            PackedFloat32Array values;
            values.resize(static_cast<int64_t>(event.embedding.size()));
            for (int64_t i = 0; i < values.size(); ++i)
                values.set(i, event.embedding[static_cast<size_t>(i)]);
            emit_signal("embedding_complete", event.request_id, session, values);
            break;
        }
        case Chorus::RuntimeEvent::Kind::Error:
            emit_signal(
                "generation_error", event.request_id, session, to_godot(event.error), to_godot_string(event.text)
            );
            break;
        case Chorus::RuntimeEvent::Kind::EngineFailed:
            emit_signal("engine_failed", to_godot(event.error), to_godot_string(event.text));
            break;
        case Chorus::RuntimeEvent::Kind::ModelLoadProgress:
            emit_signal(
                "model_load_progress",
                event.load_id.value(),
                to_godot_string(event.model_id),
                to_godot_load_phase(event.load_progress->phase),
                event.load_progress->fraction.has_value(),
                event.load_progress->fraction.value_or(0.0f)
            );
            break;
        case Chorus::RuntimeEvent::Kind::ModelLoaded:
            emit_signal("model_loaded", event.load_id.value(), to_godot_string(event.model_id));
            break;
        case Chorus::RuntimeEvent::Kind::ModelLoadFailed:
            emit_signal(
                "model_load_failed",
                event.load_id.value(),
                to_godot_string(event.model_id),
                to_godot(event.error),
                to_godot_string(event.text)
            );
            break;
        }
    }
}

// Core API

Ref<ChorusLoadResult> GodotChorus::load_model() {
    Ref<ChorusLoadResult> result;
    result.instantiate();
    if (_provider != PROVIDER_ECHO && _model_path.is_empty()) {
        result->set_result(-1, ERR_INVALID_REQUEST, "model_path is not set.");
        return result;
    }

    const Chorus::Provider provider = to_chorus_provider(_provider);
    Chorus::ChorusConfig config;
    config.log_level = effective_log_level();
    const String filesystem_path = _model_path.begins_with("res://") || _model_path.begins_with("user://")
                                       ? ProjectSettings::get_singleton()->globalize_path(_model_path)
                                       : _model_path;
    config.model = Chorus::make_initial_model_spec(
        provider,
        std::string(_model_path.get_file().get_basename().utf8().get_data()),
        std::string(filesystem_path.utf8().get_data())
    );
    // Echo accepts an empty `Chorus::InitialModelSpec` and declares no load
    // options, so this generic path contributes nothing without a special case.
    const auto& caps = provider_capabilities();
    if (!caps.load_options.empty())
        config.provider_options[caps.provider_id] = Chorus::resolve_option_defaults(caps.load_options, _load_options);

    auto engine = Chorus::make_engine(provider);
    const auto admission = _runtime.load_engine(std::move(engine), config);
    result->set_result(admission.load_id, to_godot(admission.error), to_godot_string(admission.message));
    return result;
}

#ifdef CHORUS_HOST_TEST
void GodotChorus::test_hold_next_retirement() {
    _runtime.test_hold_next_retirement();
}
bool GodotChorus::test_retirement_held() const {
    return _runtime.test_retirement_held();
}
void GodotChorus::test_release_retirement() {
    _runtime.test_release_retirement();
}
#endif

bool GodotChorus::cancel_load(int64_t load_id) {
    return _runtime.cancel_load(load_id);
}
int64_t GodotChorus::get_active_load_id() const {
    return _runtime.active_load_id().value_or(-1);
}

void GodotChorus::stop_all() {
    _runtime.stop_all();
}

bool GodotChorus::is_loaded() const {
    return _runtime.is_loaded();
}

bool GodotChorus::supports_embeddings() const {
    const auto capabilities = _runtime.capabilities();
    return capabilities && capabilities->embeddings;
}

bool GodotChorus::supports_message_token_counting() const {
    const auto capabilities = _runtime.capabilities();
    return capabilities && capabilities->message_token_counting;
}

int64_t GodotChorus::get_effective_context_size() const {
    const auto info = _runtime.loaded_model_info();
    return info ? info->per_request_context.value_or(0) : 0;
}

Ref<ChorusSubmitResult> GodotChorus::generate(const Ref<ChorusRequest>& request) {
    Ref<ChorusInferenceRequest> source = request;
    std::string error;
    if (!push_host_defaults(error))
        return rejected_submit(source, to_godot_string(error));
    auto converted = godot_chorus::generation_request_from_resource(request);
    if (std::holds_alternative<std::string>(converted))
        return rejected_submit(source, to_godot_string(std::get<std::string>(converted)));
    return submit_result(source, _runtime.submit(std::get<Chorus::GenerationRequest>(converted)));
}

Ref<ChorusSubmitResult> GodotChorus::regenerate(const Ref<ChorusRequest>& request) {
    Ref<ChorusInferenceRequest> source = request;
    std::string error;
    if (!push_host_defaults(error))
        return rejected_submit(source, to_godot_string(error));
    auto converted = godot_chorus::generation_request_from_resource(request);
    if (std::holds_alternative<std::string>(converted))
        return rejected_submit(source, to_godot_string(std::get<std::string>(converted)));
    return submit_result(source, _runtime.regenerate(std::get<Chorus::GenerationRequest>(converted)));
}

Ref<ChorusSubmitResult> GodotChorus::embed(const Ref<ChorusEmbeddingRequest>& request) {
    Ref<ChorusInferenceRequest> source = request;
    auto converted = godot_chorus::embedding_request_from_resource(request);
    if (std::holds_alternative<std::string>(converted))
        return rejected_submit(source, to_godot_string(std::get<std::string>(converted)));
    return submit_result(source, _runtime.submit(std::get<Chorus::EmbeddingRequest>(converted)));
}

TypedArray<ChorusSubmitResult> GodotChorus::generate_batch(const TypedArray<ChorusRequest>& requests) {
    TypedArray<ChorusSubmitResult> out;
    std::string error;
    if (!push_host_defaults(error)) {
        const String message = to_godot_string(error);
        for (int i = 0; i < requests.size(); ++i) {
            Ref<ChorusRequest> request = requests[i];
            Ref<ChorusInferenceRequest> source = request;
            out.push_back(rejected_submit(source, message));
        }
        return out;
    }
    std::vector<Chorus::GenerationRequest> admitted;
    std::vector<int> positions;
    out.resize(requests.size());
    for (int i = 0; i < requests.size(); ++i) {
        Ref<ChorusRequest> request = requests[i];
        auto converted = godot_chorus::generation_request_from_resource(request);
        if (std::holds_alternative<std::string>(converted)) {
            Ref<ChorusInferenceRequest> source = request;
            out[i] = rejected_submit(source, to_godot_string(std::get<std::string>(converted)));
        } else {
            positions.push_back(i);
            admitted.push_back(std::get<Chorus::GenerationRequest>(std::move(converted)));
        }
    }
    const auto results = _runtime.submit_batch(admitted);
    for (size_t j = 0; j < results.size(); ++j) {
        Ref<ChorusRequest> request = requests[positions[j]];
        Ref<ChorusInferenceRequest> source = request;
        out[positions[j]] = submit_result(source, results[j]);
    }
    return out;
}

TypedArray<ChorusSubmitResult> GodotChorus::embed_batch(const TypedArray<ChorusEmbeddingRequest>& requests) {
    TypedArray<ChorusSubmitResult> out;
    for (int i = 0; i < requests.size(); ++i) {
        Ref<ChorusEmbeddingRequest> request = requests[i];
        out.push_back(embed(request));
    }
    return out;
}

// Runtime controls

bool GodotChorus::cancel_request(int64_t request_id) {
    return _runtime.cancel(request_id);
}

bool GodotChorus::is_request_active(int64_t request_id) const {
    return _runtime.is_request_active(request_id);
}

int64_t GodotChorus::active_request_for_session(const StringName& session) const {
    auto active = _runtime.active_request_for_session(std::string(String(session).utf8().get_data()));
    return active.has_value() ? *active : -1;
}

// Conversation history

Ref<ChorusResult>
GodotChorus::import_conversation_history(const StringName& session, const TypedArray<ChorusMessage>& history) {
    std::vector<Chorus::ConversationMessage> messages;
    messages.reserve(history.size());
    for (int i = 0; i < history.size(); ++i) {
        Ref<ChorusMessage> item = history[i];
        if (item.is_null())
            return Ref<ChorusResult>(
                memnew(ChorusResult(ERR_INVALID_REQUEST, "history contains a null ChorusMessage."))
            );
        Chorus::MessageRole role;
        switch (item->get_role()) {
        case ChorusRole::SYSTEM:
            role = Chorus::MessageRole::System;
            break;
        case ChorusRole::USER:
            role = Chorus::MessageRole::User;
            break;
        case ChorusRole::ASSISTANT:
            role = Chorus::MessageRole::Assistant;
            break;
        default:
            return Ref<ChorusResult>(memnew(ChorusResult(ERR_INVALID_REQUEST, "history role is invalid.")));
        }
        messages.push_back(
            {item->get_id(), {role, Chorus::MessageContent::text(std::string(item->get_content().utf8().get_data()))}}
        );
    }
    return operation_result(
        _runtime.import_conversation_history(std::string(String(session).utf8().get_data()), std::move(messages))
    );
}

TypedArray<ChorusMessage> GodotChorus::export_conversation_history(const StringName& session) const {
    TypedArray<ChorusMessage> out;
    for (const auto& item : _runtime.export_conversation_history(std::string(String(session).utf8().get_data()))) {
        ChorusRole::Value role;
        switch (item.message.role) {
        case Chorus::MessageRole::System:
            role = ChorusRole::SYSTEM;
            break;
        case Chorus::MessageRole::User:
            role = ChorusRole::USER;
            break;
        case Chorus::MessageRole::Assistant:
            role = ChorusRole::ASSISTANT;
            break;
        default:
            continue;
        }
        const auto content = Chorus::joined_text(item.message.content);
        if (content)
            out.push_back(ChorusMessage::create(item.id, role, to_godot_string(*content)));
    }
    return out;
}

Ref<ChorusResult> GodotChorus::clear_conversation_history(const StringName& session) {
    return operation_result(_runtime.clear_conversation_history(std::string(String(session).utf8().get_data())));
}

Ref<ChorusResult> GodotChorus::edit_message(const StringName& session, int64_t message_id, const String& content) {
    return operation_result(_runtime.edit_message(
        std::string(String(session).utf8().get_data()),
        message_id,
        Chorus::MessageContent::text(std::string(content.utf8().get_data()))
    ));
}

TypedArray<StringName> GodotChorus::list_conversations() const {
    TypedArray<StringName> out;
    for (const auto& session : _runtime.list_conversations())
        out.push_back(StringName(to_godot_string(session)));
    return out;
}

Ref<ChorusResult> GodotChorus::reset_context() {
    return operation_result(_runtime.reset_context());
}

GodotChorus::TurnOutcomeCode GodotChorus::last_turn_outcome(const StringName& session) const {
    return static_cast<TurnOutcomeCode>(
        to_godot(_runtime.last_turn_outcome(std::string(String(session).utf8().get_data())))
    );
}

Ref<ChorusSubmitResult> GodotChorus::render_prompt(const Ref<ChorusRequest>& request) {
    Ref<ChorusInferenceRequest> source = request;
    std::string error;
    if (!push_host_defaults(error))
        return rejected_submit(source, to_godot_string(error));
    auto converted = godot_chorus::generation_request_from_resource(request);
    if (std::holds_alternative<std::string>(converted))
        return rejected_submit(source, to_godot_string(std::get<std::string>(converted)));
    return submit_result(source, _runtime.render_prompt(std::get<Chorus::GenerationRequest>(converted)));
}

Ref<ChorusSubmitResult> GodotChorus::count_message_tokens(const String& content) {
    const CharString utf8 = content.utf8();
    return submit_result(
        {},
        _runtime.count_message_tokens(
            Chorus::MessageContent::text(std::string(utf8.get_data(), static_cast<size_t>(utf8.length())))
        )
    );
}

// Properties

void GodotChorus::set_model_path(const String& path) {
    _model_path = path;
}
String GodotChorus::get_model_path() const {
    return _model_path;
}

// Provider load options
//
// The selected provider declares its schema. These hooks render that schema,
// so the adapter names no provider option, default, or range.

static bool
is_prerequisite_for_any_option(const Chorus::ProviderOptionDescriptors& declared_options, const std::string& key) {
    for (const auto& option : declared_options) {
        if (option.prerequisite_option && *option.prerequisite_option == key)
            return true;
    }
    return false;
}

const Chorus::EngineCapabilities& GodotChorus::provider_capabilities() const {
    if (_cached_capabilities_provider != _provider) {
        _cached_capabilities = Chorus::describe_provider_capabilities(to_chorus_provider(_provider));
        _cached_capabilities_provider = _provider;
    }
    return _cached_capabilities;
}

const Chorus::ProviderOptionDescriptors& GodotChorus::load_option_descriptors() const {
    return provider_capabilities().load_options;
}

const Chorus::ProviderOptionDescriptor* GodotChorus::find_load_option(const StringName& name) const {
    return Chorus::find_option_descriptor(load_option_descriptors(), std::string(String(name).utf8().get_data()));
}

bool GodotChorus::_set(const StringName& name, const Variant& value) {
    const auto* descriptor = find_load_option(name);
    if (!descriptor)
        return false;
    auto coerced = godot_chorus::coerce_to_descriptor(*descriptor, value);
    if (!coerced) {
        UtilityFunctions::push_error(
            String("[Chorus] ") + to_godot_string(descriptor->key) + String(": wrong value type for this option.")
        );
        // Godot expects true because the property exists, even though its value
        // could not be accepted.
        return true;
    }
    _load_options[descriptor->key] = std::move(*coerced);
    if (is_loaded() || get_active_load_id() >= 0)
        UtilityFunctions::push_warning(
            "[Chorus] load option changed while loaded or loading; takes effect on the next accepted load_model()."
        );
    if (is_prerequisite_for_any_option(load_option_descriptors(), descriptor->key)) {
        // Dependent options may have entered or left the Inspector surface.
        notify_property_list_changed();
    }
    return true;
}

bool GodotChorus::_get(const StringName& name, Variant& ret) const {
    const auto* descriptor = find_load_option(name);
    if (!descriptor)
        return false;
    const auto stored = _load_options.find(descriptor->key);
    ret = godot_chorus::option_value_to_variant(
        stored != _load_options.end() ? stored->second : descriptor->default_value
    );
    return true;
}

void GodotChorus::_get_property_list(List<PropertyInfo>* list) const {
    const auto& descriptors = load_option_descriptors();
    if (descriptors.empty())
        return;
    list->push_back(PropertyInfo(Variant::NIL, "Provider Options", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
    for (const auto& descriptor : descriptors) {
        if (descriptor.presentation != Chorus::ProviderOptionPresentation::Normal)
            continue;
        const bool enabled = Chorus::is_prerequisite_option_enabled(descriptors, descriptor, _load_options);
        list->push_back(godot_chorus::property_info_for(descriptor, enabled));
    }
    list->push_back(
        PropertyInfo(Variant::NIL, "Advanced Provider Options", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP)
    );
    for (const auto& descriptor : descriptors) {
        if (descriptor.presentation != Chorus::ProviderOptionPresentation::Advanced)
            continue;
        const bool enabled = Chorus::is_prerequisite_option_enabled(descriptors, descriptor, _load_options);
        list->push_back(godot_chorus::property_info_for(descriptor, enabled));
    }
}

bool GodotChorus::_property_can_revert(const StringName& name) const {
    return find_load_option(name) != nullptr;
}

bool GodotChorus::_property_get_revert(const StringName& name, Variant& ret) const {
    const auto* descriptor = find_load_option(name);
    if (!descriptor)
        return false;
    ret = godot_chorus::option_value_to_variant(descriptor->default_value);
    return true;
}

void GodotChorus::set_provider(ProviderChoice provider) {
    if (is_loaded() || get_active_load_id() >= 0) {
        UtilityFunctions::push_warning(
            "[Chorus] provider changed while loaded or loading; takes effect on the next accepted load_model()."
        );
    }
    _provider = provider;
    // The new provider may expose a different Inspector surface.
    notify_property_list_changed();
}

GodotChorus::ProviderChoice GodotChorus::get_provider() const {
    return _provider;
}

void GodotChorus::set_override_log_level(bool enabled) {
    _override_log_level = enabled;
}
bool GodotChorus::get_override_log_level() const {
    return _override_log_level;
}
void GodotChorus::set_log_level(int64_t level) {
    _log_level = level;
}
int64_t GodotChorus::get_log_level() const {
    return _log_level;
}

bool GodotChorus::push_host_defaults(std::string& error) {
    Chorus::GenerationDefaults defaults;
    if (!godot_chorus::project_generation_defaults(defaults, error)) {
        UtilityFunctions::push_error("[Chorus] " + to_godot_string(error));
        return false;
    }
    _runtime.set_generation_defaults(std::move(defaults));
    return true;
}

// Utility

float GodotChorus::similarity_cos(PackedFloat32Array array1, PackedFloat32Array array2) const {
    if (array1.size() != array2.size() || array1.is_empty()) {
        UtilityFunctions::push_error("[Chorus] similarity_cos: arrays must be non-empty and the same size.");
        return 0.0f;
    }
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < array1.size(); ++i) {
        dot += (double)array1[i] * (double)array2[i];
        norm1 += (double)array1[i] * (double)array1[i];
        norm2 += (double)array2[i] * (double)array2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0)
        return 0.0f;
    return (float)(dot / (std::sqrt(norm1) * std::sqrt(norm2)));
}

// Godot bindings

void GodotChorus::_bind_methods() {
    // Signals
    ADD_SIGNAL(MethodInfo(
        "model_load_progress",
        PropertyInfo(Variant::INT, "load_id"),
        PropertyInfo(Variant::STRING, "model_id"),
        PropertyInfo(Variant::INT, "phase", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusLoadPhase.Value"),
        PropertyInfo(Variant::BOOL, "has_fraction"),
        PropertyInfo(Variant::FLOAT, "fraction")
    ));
    ADD_SIGNAL(
        MethodInfo("model_loaded", PropertyInfo(Variant::INT, "load_id"), PropertyInfo(Variant::STRING, "model_id"))
    );
    ADD_SIGNAL(MethodInfo(
        "model_load_failed",
        PropertyInfo(Variant::INT, "load_id"),
        PropertyInfo(Variant::STRING, "model_id"),
        PropertyInfo(Variant::INT, "error_code"),
        PropertyInfo(Variant::STRING, "message")
    ));
    ADD_SIGNAL(MethodInfo(
        "token_generated",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::STRING, "token")
    ));
    ADD_SIGNAL(MethodInfo(
        "reasoning_token_generated",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::STRING, "token")
    ));
    ADD_SIGNAL(MethodInfo(
        "generation_complete",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::INT, "message_id"),
        PropertyInfo(Variant::STRING, "content"),
        PropertyInfo(Variant::STRING, "reasoning")
    ));
    ADD_SIGNAL(MethodInfo(
        "embedding_complete",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "embedding")
    ));
    ADD_SIGNAL(MethodInfo(
        "history_truncated",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::PACKED_INT64_ARRAY, "omitted_message_ids")
    ));
    ADD_SIGNAL(MethodInfo(
        "generation_error",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::INT, "error_code"),
        PropertyInfo(Variant::STRING, "message")
    ));
    ADD_SIGNAL(
        MethodInfo("engine_failed", PropertyInfo(Variant::INT, "error_code"), PropertyInfo(Variant::STRING, "message"))
    );
    // One diagnostic as the provider stated it: a stable message plus typed
    // fields, allowing projects to group and filter without parsing prose.
    // `request_id` is `-1` and `session` is empty when no work owns the record.
    ADD_SIGNAL(MethodInfo(
        "log_record",
        PropertyInfo(Variant::INT, "level"),
        PropertyInfo(Variant::STRING, "message"),
        PropertyInfo(Variant::DICTIONARY, "fields"),
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::FLOAT, "produced_at")
    ));

    ADD_SIGNAL(MethodInfo(
        "prompt_rendered",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING_NAME, "session"),
        PropertyInfo(Variant::STRING, "text"),
        PropertyInfo(Variant::PACKED_INT64_ARRAY, "omitted_message_ids")
    ));
    ADD_SIGNAL(MethodInfo(
        "message_token_counted", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::INT, "token_count")
    ));

    // `GodotChorus::ErrorCode`
    BIND_ENUM_CONSTANT(ERR_NONE);
    BIND_ENUM_CONSTANT(ERR_MODEL_LOAD);
    BIND_ENUM_CONSTANT(ERR_CONTEXT_INIT);
    BIND_ENUM_CONSTANT(ERR_DECODE);
    BIND_ENUM_CONSTANT(ERR_TOKENIZE);
    BIND_ENUM_CONSTANT(ERR_INVALID_REQUEST);
    BIND_ENUM_CONSTANT(ERR_ENGINE_NOT_READY);
    BIND_ENUM_CONSTANT(ERR_CANCELLED);
    BIND_ENUM_CONSTANT(ERR_UNSUPPORTED_MODEL_FORMAT);
    BIND_ENUM_CONSTANT(ERR_UNSUPPORTED_FEATURE);
    BIND_ENUM_CONSTANT(ERR_UNSUPPORTED_OPTION);
    BIND_ENUM_CONSTANT(ERR_SESSION_BUSY);
    BIND_ENUM_CONSTANT(ERR_UNKNOWN);

    // `GodotChorus::ProviderChoice`
    BIND_ENUM_CONSTANT(PROVIDER_LLAMA);
    BIND_ENUM_CONSTANT(PROVIDER_ECHO);

    // `GodotChorus::TurnOutcomeCode`
    BIND_ENUM_CONSTANT(TURN_NONE);
    BIND_ENUM_CONSTANT(TURN_COMPLETED);
    BIND_ENUM_CONSTANT(TURN_CANCELLED);
    BIND_ENUM_CONSTANT(TURN_ERRORED);

    // `GodotChorus::LogLevelCode`
    BIND_ENUM_CONSTANT(LOG_DEBUG);
    BIND_ENUM_CONSTANT(LOG_INFO);
    BIND_ENUM_CONSTANT(LOG_WARN);
    BIND_ENUM_CONSTANT(LOG_ERROR);
    BIND_ENUM_CONSTANT(LOG_FATAL);
    BIND_ENUM_CONSTANT(LOG_OFF);

    // Runtime methods
    ClassDB::bind_method(D_METHOD("load_model"), &GodotChorus::load_model);
#ifdef CHORUS_HOST_TEST
    ClassDB::bind_method(D_METHOD("test_hold_next_retirement"), &GodotChorus::test_hold_next_retirement);
    ClassDB::bind_method(D_METHOD("test_retirement_held"), &GodotChorus::test_retirement_held);
    ClassDB::bind_method(D_METHOD("test_release_retirement"), &GodotChorus::test_release_retirement);
#endif
    ClassDB::bind_method(D_METHOD("cancel_load", "load_id"), &GodotChorus::cancel_load);
    ClassDB::bind_method(D_METHOD("get_active_load_id"), &GodotChorus::get_active_load_id);
    ClassDB::bind_method(D_METHOD("stop_all"), &GodotChorus::stop_all);
    ClassDB::bind_method(D_METHOD("is_loaded"), &GodotChorus::is_loaded);
    ClassDB::bind_method(D_METHOD("supports_embeddings"), &GodotChorus::supports_embeddings);
    ClassDB::bind_method(D_METHOD("generate", "request"), &GodotChorus::generate);
    ClassDB::bind_method(D_METHOD("generate_batch", "requests"), &GodotChorus::generate_batch);
    ClassDB::bind_method(D_METHOD("regenerate", "request"), &GodotChorus::regenerate);
    ClassDB::bind_method(D_METHOD("embed", "request"), &GodotChorus::embed);
    ClassDB::bind_method(D_METHOD("embed_batch", "requests"), &GodotChorus::embed_batch);
    ClassDB::bind_method(D_METHOD("cancel_request", "request_id"), &GodotChorus::cancel_request);
    ClassDB::bind_method(D_METHOD("is_request_active", "request_id"), &GodotChorus::is_request_active);
    ClassDB::bind_method(D_METHOD("active_request_for_session", "session"), &GodotChorus::active_request_for_session);

    ClassDB::bind_method(
        D_METHOD("import_conversation_history", "session", "history"), &GodotChorus::import_conversation_history
    );
    ClassDB::bind_method(D_METHOD("export_conversation_history", "session"), &GodotChorus::export_conversation_history);
    ClassDB::bind_method(D_METHOD("clear_conversation_history", "session"), &GodotChorus::clear_conversation_history);
    ClassDB::bind_method(D_METHOD("edit_message", "session", "message_id", "content"), &GodotChorus::edit_message);
    ClassDB::bind_method(D_METHOD("list_conversations"), &GodotChorus::list_conversations);
    ClassDB::bind_method(D_METHOD("reset_context"), &GodotChorus::reset_context);
    ClassDB::bind_method(D_METHOD("last_turn_outcome", "session"), &GodotChorus::last_turn_outcome);
    ClassDB::bind_method(D_METHOD("render_prompt", "request"), &GodotChorus::render_prompt);
    ClassDB::bind_method(D_METHOD("count_message_tokens", "content"), &GodotChorus::count_message_tokens);
    ClassDB::bind_method(D_METHOD("supports_message_token_counting"), &GodotChorus::supports_message_token_counting);

    // Utility methods
    ClassDB::bind_method(D_METHOD("similarity_cos", "array1", "array2"), &GodotChorus::similarity_cos);

    // Properties
    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &GodotChorus::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &GodotChorus::get_model_path);
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"), "set_model_path", "get_model_path"
    );
    ClassDB::bind_method(D_METHOD("get_effective_context_size"), &GodotChorus::get_effective_context_size);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "effective_context_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY),
        "",
        "get_effective_context_size"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "active_load_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY),
        "",
        "get_active_load_id"
    );

    // Provider load options such as `context_size` and `use_gpu` are not bound
    // here. `GodotChorus::_get_property_list` renders the selected provider's
    // declaration, so the adapter never restates its schema.

    ClassDB::bind_method(D_METHOD("set_provider", "provider"), &GodotChorus::set_provider);
    ClassDB::bind_method(D_METHOD("get_provider"), &GodotChorus::get_provider);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "provider", PROPERTY_HINT_ENUM, "Llama,Echo"), "set_provider", "get_provider"
    );

    ClassDB::bind_method(D_METHOD("set_override_log_level", "enabled"), &GodotChorus::set_override_log_level);
    ClassDB::bind_method(D_METHOD("get_override_log_level"), &GodotChorus::get_override_log_level);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_log_level"), "set_override_log_level", "get_override_log_level");
    ClassDB::bind_method(D_METHOD("set_log_level", "level"), &GodotChorus::set_log_level);
    ClassDB::bind_method(D_METHOD("get_log_level"), &GodotChorus::get_log_level);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "log_level", PROPERTY_HINT_ENUM, LOG_LEVEL_HINT), "set_log_level", "get_log_level"
    );
}
