#include "godot_chorus/godot_chorus.hpp"

#include <cmath>
#include <memory>
#include <variant>

#include "chorus/engine_factory.hpp"
#include "godot_chorus/generation_request_normalizer.hpp"
#include "godot_chorus/provider_option_properties.hpp"

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

static Chorus::Provider to_chorus_provider(GodotChorus::ProviderChoice provider) {
    return provider == GodotChorus::PROVIDER_ECHO ? Chorus::Provider::Echo : Chorus::Provider::Llama;
}

static bool gates_another_option(const Chorus::ProviderOptionDescriptors& schema, const std::string& key) {
    for (const auto& descriptor : schema) {
        if (descriptor.enabled_by && *descriptor.enabled_by == key)
            return true;
    }
    return false;
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

// ===========================================================================
// Lifecycle
// ===========================================================================

void GodotChorus::_notification(int p_what) {
    if (p_what == NOTIFICATION_READY) {
        set_process(true);
    }
}

void GodotChorus::_process(double /*delta*/) {
    for (const auto& event : _runtime.poll()) {
        const String session = event.session_id ? String(event.session_id->c_str()) : String();
        switch (event.kind) {
        case Chorus::RuntimeEvent::Kind::Token:
            emit_signal("token_generated", event.request_id, session, String(event.text.c_str()));
            break;
        case Chorus::RuntimeEvent::Kind::ReasoningToken:
            emit_signal("reasoning_token_generated", event.request_id, session, String(event.text.c_str()));
            break;
        case Chorus::RuntimeEvent::Kind::HistoryTruncated:
            emit_signal("history_truncated", session, event.dropped);
            break;
        case Chorus::RuntimeEvent::Kind::Complete:
            emit_signal(
                "generation_complete",
                event.request_id,
                session,
                String(event.text.c_str()),
                String(event.reasoning.c_str())
            );
            break;
        case Chorus::RuntimeEvent::Kind::Error:
            emit_signal(
                "generation_error", event.request_id, session, to_godot(event.error), String(event.text.c_str())
            );
            break;
        case Chorus::RuntimeEvent::Kind::EngineFailed:
            emit_signal("engine_failed", to_godot(event.error), String(event.text.c_str()));
            break;
        }
    }
}

// ===========================================================================
// Core API
// ===========================================================================

bool GodotChorus::load_model() {
    if (_provider != PROVIDER_ECHO && _model_path.is_empty()) {
        UtilityFunctions::push_error("[Chorus] model_path is not set.");
        return false;
    }

    Chorus::ChorusConfig config;
    config.log_callback = [](Chorus::LogLevel level, const std::string& msg) {
        String godot_msg = String("[Chorus] ") + String(msg.c_str());
        switch (level) {
        case Chorus::LogLevel::Debug:
        case Chorus::LogLevel::Info:
            UtilityFunctions::print(godot_msg);
            break;
        case Chorus::LogLevel::Warn:
            UtilityFunctions::push_warning(godot_msg);
            break;
        case Chorus::LogLevel::Error:
        case Chorus::LogLevel::Fatal:
            UtilityFunctions::push_error(godot_msg);
            break;
        }
    };
    if (_provider != PROVIDER_ECHO) {
        config.model.model_id = _model_path.get_file().get_basename().utf8().get_data();
        config.model.format = Chorus::ModelFormat::Gguf;
        config.model.assets.push_back({Chorus::AssetRole::Weights, _model_path.utf8().get_data()});
    }
    // Echo: empty InitialModelSpec, and it declares no load options, so the loop below
    // contributes nothing rather than needing a special case.
    const auto& caps = provider_capabilities();
    if (!caps.load_options.empty())
        config.provider_options[caps.provider_id] = Chorus::resolve_option_defaults(caps.load_options, _load_options);

    auto engine = Chorus::make_engine(to_chorus_provider(_provider));
    auto err = _runtime.load_engine(std::move(engine), config);
    if (err.has_value()) {
        UtilityFunctions::push_error(String("[Chorus] Model load failed: ") + chorus_error_name(err.value()));
        return false;
    }
    return true;
}

void GodotChorus::stop_all() {
    _runtime.stop_all();
}

bool GodotChorus::is_loaded() const {
    return _runtime.is_loaded();
}

int64_t GodotChorus::generate(const Dictionary& request) {
    push_host_defaults();

    auto normalized = godot_chorus::normalize_generation_request(request);
    if (std::holds_alternative<String>(normalized)) {
        UtilityFunctions::push_error(std::get<String>(normalized));
        return -1;
    }

    auto& gen_request = std::get<Chorus::GenerationRequest>(normalized);
    auto result = _runtime.submit(gen_request);
    if (!result.ok()) {
        String message = String("[Chorus] generate() rejected: ") + chorus_error_name(result.error);
        if (!result.message.empty())
            message += String(" - ") + String(result.message.c_str());
        if (result.error == Chorus::ChorusError::EngineNotReady)
            message += ". Call load_model() first.";
        UtilityFunctions::push_error(message);
        return -1;
    }
    return result.request_id;
}

int64_t GodotChorus::regenerate(const String& session, const Dictionary& overrides) {
    push_host_defaults();

    auto normalized = godot_chorus::normalize_generation_overrides(overrides);
    if (std::holds_alternative<String>(normalized)) {
        UtilityFunctions::push_error(std::get<String>(normalized));
        return -1;
    }

    auto& gen_request = std::get<Chorus::GenerationRequest>(normalized);
    gen_request.session_id = std::string(session.utf8().get_data());
    auto result = _runtime.regenerate(gen_request);
    if (!result.ok()) {
        String message = String("[Chorus] regenerate() rejected: ") + chorus_error_name(result.error);
        if (!result.message.empty())
            message += String(" - ") + String(result.message.c_str());
        if (result.error == Chorus::ChorusError::EngineNotReady)
            message += ". Call load_model() first.";
        UtilityFunctions::push_error(message);
        return -1;
    }
    return result.request_id;
}

// ===========================================================================
// Runtime controls
// ===========================================================================

bool GodotChorus::cancel_request(int64_t request_id) {
    return _runtime.cancel(request_id);
}

bool GodotChorus::is_request_active(int64_t request_id) const {
    return _runtime.is_request_active(request_id);
}

int64_t GodotChorus::active_request_for_session(const String& session) const {
    auto active = _runtime.active_request_for_session(std::string(session.utf8().get_data()));
    return active.has_value() ? *active : -1;
}

// ===========================================================================
// Conversation history
// ===========================================================================

static Dictionary chat_message_to_dict(const Chorus::ChatMessage& message) {
    Dictionary dict;
    dict["role"] = String(message.role.c_str());
    dict["content"] = String(message.content.c_str());
    return dict;
}

bool GodotChorus::import_conversation_history(const String& session, const Array& history) {
    std::vector<Chorus::ChatMessage> messages;
    messages.reserve(history.size());
    for (int i = 0; i < history.size(); ++i) {
        if (history[i].get_type() != Variant::DICTIONARY) {
            UtilityFunctions::push_error(
                "[Chorus] import_conversation_history: entries must be {role, content} Dictionaries."
            );
            return false;
        }
        const Dictionary entry = history[i];
        if (!entry.has("role") || !entry.has("content")) {
            UtilityFunctions::push_error("[Chorus] import_conversation_history: entries require 'role' and 'content'.");
            return false;
        }
        if (entry["role"].get_type() != Variant::STRING || entry["content"].get_type() != Variant::STRING) {
            UtilityFunctions::push_error("[Chorus] import_conversation_history: 'role' and 'content' must be Strings.");
            return false;
        }
        messages.push_back(
            {std::string(((String)entry["role"]).utf8().get_data()),
             std::string(((String)entry["content"]).utf8().get_data())}
        );
    }
    auto err = _runtime.import_conversation_history(std::string(session.utf8().get_data()), std::move(messages));
    if (err.has_value()) {
        UtilityFunctions::push_error(String("[Chorus] import_conversation_history failed: ") + chorus_error_name(*err));
        return false;
    }
    return true;
}

Array GodotChorus::export_conversation_history(const String& session) const {
    Array out;
    for (const auto& message : _runtime.export_conversation_history(std::string(session.utf8().get_data())))
        out.push_back(chat_message_to_dict(message));
    return out;
}

bool GodotChorus::clear_conversation_history(const String& session) {
    auto err = _runtime.clear_conversation_history(std::string(session.utf8().get_data()));
    if (err.has_value()) {
        UtilityFunctions::push_error(String("[Chorus] clear_conversation_history failed: ") + chorus_error_name(*err));
        return false;
    }
    return true;
}

bool GodotChorus::edit_message(const String& session, int64_t index, const String& content) {
    auto err =
        _runtime.edit_message(std::string(session.utf8().get_data()), index, std::string(content.utf8().get_data()));
    if (err.has_value()) {
        UtilityFunctions::push_error(String("[Chorus] edit_message failed: ") + chorus_error_name(*err));
        return false;
    }
    return true;
}

PackedStringArray GodotChorus::list_conversations() const {
    PackedStringArray out;
    for (const auto& session : _runtime.list_conversations())
        out.push_back(String(session.c_str()));
    return out;
}

bool GodotChorus::reset_context() {
    auto err = _runtime.reset_context();
    if (err.has_value()) {
        UtilityFunctions::push_error(
            String("[Chorus] reset_context failed: ") + chorus_error_name(*err) +
            String(" - cancel or stop_all + poll first.")
        );
        return false;
    }
    return true;
}

GodotChorus::TurnOutcomeCode GodotChorus::last_turn_outcome(const String& session) const {
    return static_cast<TurnOutcomeCode>(to_godot(_runtime.last_turn_outcome(std::string(session.utf8().get_data()))));
}

String GodotChorus::render_chat_prompt(const String& session, const String& template_override, const Array& inject) {
    auto parsed = godot_chorus::normalize_inject_array(inject);
    if (std::holds_alternative<String>(parsed)) {
        UtilityFunctions::push_error(std::get<String>(parsed));
        return String();
    }

    push_host_defaults();
    auto rendered = _runtime.render_prompt(
        std::string(session.utf8().get_data()),
        std::string(template_override.utf8().get_data()),
        std::get<std::vector<Chorus::InjectedMessage>>(parsed)
    );
    return rendered.has_value() ? String(rendered->c_str()) : String();
}

// ===========================================================================
// Properties
// ===========================================================================

void GodotChorus::set_model_path(const String& path) {
    _model_path = path;
}
String GodotChorus::get_model_path() const {
    return _model_path;
}

// ---------------------------------------------------------------------------
// Provider load options
//
// The selected provider declares its own option schema; these hooks render it.
// Nothing here names an option, a default, or a range: swap the provider and the
// inspector follows without a line of adapter code changing.
// ---------------------------------------------------------------------------

const Chorus::EngineCapabilities& GodotChorus::provider_capabilities() const {
    if (_cached_capabilities_provider != _provider) {
        _cached_capabilities = Chorus::describe_provider(to_chorus_provider(_provider));
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
            String("[Chorus] ") + String(descriptor->key.c_str()) + String(": wrong value type for this option.")
        );
        return true; // handled: the property exists, the value did not fit
    }
    _load_options[descriptor->key] = std::move(*coerced);
    if (gates_another_option(load_option_descriptors(), descriptor->key))
        notify_property_list_changed(); // the options it governs just changed state
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
        const bool enabled = Chorus::option_is_enabled(descriptors, descriptor, _load_options);
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
    if (is_loaded()) {
        UtilityFunctions::push_warning(
            "[Chorus] provider changed while loaded; takes effect on the next load_model()."
        );
    }
    _provider = provider;
    notify_property_list_changed(); // a different provider declares different options
}

GodotChorus::ProviderChoice GodotChorus::get_provider() const {
    return _provider;
}

void GodotChorus::set_generation_defaults(const Ref<ChorusGenerationDefaults>& defaults) {
    _generation_defaults = defaults;
}

Ref<ChorusGenerationDefaults> GodotChorus::get_generation_defaults() const {
    return _generation_defaults;
}

Ref<ChorusGenerationDefaults> GodotChorus::effective_generation_defaults() {
    if (_generation_defaults.is_valid())
        return _generation_defaults;
    if (_fallback_generation_defaults.is_null())
        _fallback_generation_defaults.instantiate();
    return _fallback_generation_defaults;
}

void GodotChorus::set_chat_template(const String& chat_template) {
    _chat_template = chat_template;
}
String GodotChorus::get_chat_template() const {
    return _chat_template;
}

void GodotChorus::push_host_defaults() {
    // Rebuilt per call rather than pushed from the setters: the assigned
    // ChorusGenerationDefaults is a Resource a script may edit in place, and
    // the node gets no notification when it does.
    _runtime.set_host_defaults(
        {effective_generation_defaults()->to_patch(), std::string(_chat_template.utf8().get_data())}
    );
}

// ===========================================================================
// Utility
// ===========================================================================

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

// ===========================================================================
// Bindings
// ===========================================================================

void GodotChorus::_bind_methods() {
    // --- Signals ---
    ADD_SIGNAL(MethodInfo(
        "token_generated",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING, "session"),
        PropertyInfo(Variant::STRING, "token")
    ));
    ADD_SIGNAL(MethodInfo(
        "reasoning_token_generated",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING, "session"),
        PropertyInfo(Variant::STRING, "token")
    ));
    ADD_SIGNAL(MethodInfo(
        "generation_complete",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING, "session"),
        PropertyInfo(Variant::STRING, "full_text"),
        PropertyInfo(Variant::STRING, "reasoning")
    ));
    ADD_SIGNAL(
        MethodInfo("history_truncated", PropertyInfo(Variant::STRING, "session"), PropertyInfo(Variant::INT, "dropped"))
    );
    ADD_SIGNAL(MethodInfo(
        "generation_error",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING, "session"),
        PropertyInfo(Variant::INT, "error_code"),
        PropertyInfo(Variant::STRING, "message")
    ));
    ADD_SIGNAL(
        MethodInfo("engine_failed", PropertyInfo(Variant::INT, "error_code"), PropertyInfo(Variant::STRING, "message"))
    );

    // --- ErrorCode enum ---
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

    // --- ProviderChoice enum ---
    BIND_ENUM_CONSTANT(PROVIDER_LLAMA);
    BIND_ENUM_CONSTANT(PROVIDER_ECHO);

    // --- TurnOutcomeCode enum ---
    BIND_ENUM_CONSTANT(TURN_NONE);
    BIND_ENUM_CONSTANT(TURN_COMPLETED);
    BIND_ENUM_CONSTANT(TURN_CANCELLED);
    BIND_ENUM_CONSTANT(TURN_ERRORED);

    // --- Core methods ---
    ClassDB::bind_method(D_METHOD("load_model"), &GodotChorus::load_model);
    ClassDB::bind_method(D_METHOD("stop_all"), &GodotChorus::stop_all);
    ClassDB::bind_method(D_METHOD("is_loaded"), &GodotChorus::is_loaded);
    ClassDB::bind_method(D_METHOD("generate", "request"), &GodotChorus::generate);
    ClassDB::bind_method(D_METHOD("cancel_request", "request_id"), &GodotChorus::cancel_request);
    ClassDB::bind_method(D_METHOD("is_request_active", "request_id"), &GodotChorus::is_request_active);
    ClassDB::bind_method(D_METHOD("active_request_for_session", "session"), &GodotChorus::active_request_for_session);
    ClassDB::bind_method(D_METHOD("similarity_cos", "array1", "array2"), &GodotChorus::similarity_cos);

    // --- Conversation history ---
    ClassDB::bind_method(
        D_METHOD("regenerate", "session", "overrides"), &GodotChorus::regenerate, DEFVAL(Dictionary())
    );
    ClassDB::bind_method(
        D_METHOD("import_conversation_history", "session", "history"), &GodotChorus::import_conversation_history
    );
    ClassDB::bind_method(D_METHOD("export_conversation_history", "session"), &GodotChorus::export_conversation_history);
    ClassDB::bind_method(D_METHOD("clear_conversation_history", "session"), &GodotChorus::clear_conversation_history);
    ClassDB::bind_method(D_METHOD("edit_message", "session", "index", "content"), &GodotChorus::edit_message);
    ClassDB::bind_method(D_METHOD("list_conversations"), &GodotChorus::list_conversations);
    ClassDB::bind_method(D_METHOD("reset_context"), &GodotChorus::reset_context);
    ClassDB::bind_method(D_METHOD("last_turn_outcome", "session"), &GodotChorus::last_turn_outcome);
    ClassDB::bind_method(
        D_METHOD("render_chat_prompt", "session", "template", "inject"),
        &GodotChorus::render_chat_prompt,
        DEFVAL(String()),
        DEFVAL(Array())
    );

    // --- Properties ---
    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &GodotChorus::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &GodotChorus::get_model_path);
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"), "set_model_path", "get_model_path"
    );

    // Provider load options (context_size, use_gpu, ...) are not bound here:
    // the selected provider declares them and _get_property_list renders that
    // declaration, so the adapter never restates a provider's option schema.

    ClassDB::bind_method(D_METHOD("set_provider", "provider"), &GodotChorus::set_provider);
    ClassDB::bind_method(D_METHOD("get_provider"), &GodotChorus::get_provider);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "provider", PROPERTY_HINT_ENUM, "Llama,Echo"), "set_provider", "get_provider"
    );

    ClassDB::bind_method(D_METHOD("set_generation_defaults", "defaults"), &GodotChorus::set_generation_defaults);
    ClassDB::bind_method(D_METHOD("get_generation_defaults"), &GodotChorus::get_generation_defaults);
    ADD_PROPERTY(
        PropertyInfo(Variant::OBJECT, "generation_defaults", PROPERTY_HINT_RESOURCE_TYPE, "ChorusGenerationDefaults"),
        "set_generation_defaults",
        "get_generation_defaults"
    );

    ClassDB::bind_method(D_METHOD("set_chat_template", "chat_template"), &GodotChorus::set_chat_template);
    ClassDB::bind_method(D_METHOD("get_chat_template"), &GodotChorus::get_chat_template);
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "chat_template", PROPERTY_HINT_MULTILINE_TEXT, ""),
        "set_chat_template",
        "get_chat_template"
    );
}
