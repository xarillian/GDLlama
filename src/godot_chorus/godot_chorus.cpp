#include "godot_chorus/godot_chorus.hpp"

#include <cmath>
#include <memory>
#include <variant>

#include "chorus/engine_factory.hpp"
#include "godot_chorus/generation_request_normalizer.hpp"

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
        }
    }
}

// ===========================================================================
// Core API
// ===========================================================================

bool GodotChorus::load_model() {
    if (_backend != BACKEND_ECHO && _model_path.is_empty()) {
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
    if (_backend != BACKEND_ECHO) {
        config.model.model_id = _model_path.get_file().get_basename().utf8().get_data();
        config.model.format = Chorus::ModelFormat::Gguf;
        config.model.assets.push_back({"weights", _model_path.utf8().get_data(), std::nullopt, std::nullopt});
        Chorus::GodotAdapter::LlamaLoadOptions options;
        options.context_size = _context_size;
        options.thread_count = _thread_count;
        options.use_gpu = _use_gpu;
        options.gpu_layers = _gpu_layers;
        options.num_slots = _num_slots;
        options.tokens_per_tick = _tokens_per_tick;
        options.n_batch = _n_batch;
        options.n_ubatch = _n_ubatch;
        options.main_gpu = _main_gpu;
        config.backend_options["llama"] = Chorus::GodotAdapter::make_llama_load_options(options);
    }
    // Echo: empty ModelSpec, no options (it rejects any it is given).

    auto engine = Chorus::make_engine(_backend == BACKEND_ECHO ? Chorus::Backend::Echo : Chorus::Backend::Llama);
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
    const Chorus::GenerationConfig defaults =
        Chorus::apply_generation_patch(Chorus::GenerationConfig{}, effective_generation_defaults()->to_patch());

    auto normalized = godot_chorus::normalize_generation_request(request, defaults);
    if (std::holds_alternative<String>(normalized)) {
        UtilityFunctions::push_error(std::get<String>(normalized));
        return -1;
    }

    auto& gen_request = std::get<Chorus::GenerationRequest>(normalized);
    // Ambient chat controls apply to chat turns only: the node template and a
    // defaults-resource thinking value must not turn a stateless raw-prompt
    // call into an InvalidRequest (the runtime rejects stateless chat
    // controls). Explicit request keys still flow through and earn the
    // runtime's honest rejection.
    if (gen_request.session_id.has_value())
        gen_request.chat_template = resolve_chat_template(std::move(gen_request.chat_template));
    else if (!request.has("thinking"))
        gen_request.config.common.thinking.reset();
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
    const Chorus::GenerationConfig defaults =
        Chorus::apply_generation_patch(Chorus::GenerationConfig{}, effective_generation_defaults()->to_patch());

    auto normalized = godot_chorus::normalize_generation_overrides(overrides, defaults);
    if (std::holds_alternative<String>(normalized)) {
        UtilityFunctions::push_error(std::get<String>(normalized));
        return -1;
    }

    auto& gen_request = std::get<Chorus::GenerationRequest>(normalized);
    gen_request.session_id = std::string(session.utf8().get_data());
    gen_request.chat_template = resolve_chat_template(std::move(gen_request.chat_template));
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
    const std::string session_id(session.utf8().get_data());
    auto history = _runtime.export_conversation_history(session_id);
    if (history.empty()) {
        UtilityFunctions::push_error("[Chorus] edit_message: unknown session or empty history.");
        return false;
    }
    const int64_t resolved = index < 0 ? (int64_t)history.size() + index : index;
    if (resolved < 0 || resolved >= (int64_t)history.size()) {
        UtilityFunctions::push_error("[Chorus] edit_message: index out of range.");
        return false;
    }
    history[resolved].content = std::string(content.utf8().get_data());
    auto err = _runtime.import_conversation_history(session_id, std::move(history));
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

    const Chorus::GenerationConfig defaults =
        Chorus::apply_generation_patch(Chorus::GenerationConfig{}, effective_generation_defaults()->to_patch());
    auto rendered = _runtime.render_prompt(
        std::string(session.utf8().get_data()),
        resolve_chat_template(std::string(template_override.utf8().get_data())),
        std::get<std::vector<Chorus::InjectedMessage>>(parsed),
        defaults
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

void GodotChorus::set_context_size(int32_t size) {
    _context_size = size;
}
int32_t GodotChorus::get_context_size() const {
    return _context_size;
}

void GodotChorus::set_thread_count(int32_t count) {
    _thread_count = count;
}
int32_t GodotChorus::get_thread_count() const {
    return _thread_count;
}

void GodotChorus::set_use_gpu(bool use) {
    _use_gpu = use;
}
bool GodotChorus::get_use_gpu() const {
    return _use_gpu;
}

void GodotChorus::set_gpu_layers(int32_t layers) {
    _gpu_layers = layers;
}
int32_t GodotChorus::get_gpu_layers() const {
    return _gpu_layers;
}

void GodotChorus::set_num_slots(int32_t count) {
    _num_slots = count;
}
int32_t GodotChorus::get_num_slots() const {
    return _num_slots;
}

void GodotChorus::set_tokens_per_tick(int32_t count) {
    _tokens_per_tick = count;
}
int32_t GodotChorus::get_tokens_per_tick() const {
    return _tokens_per_tick;
}

void GodotChorus::set_n_batch(int32_t count) {
    _n_batch = count;
}
int32_t GodotChorus::get_n_batch() const {
    return _n_batch;
}

void GodotChorus::set_n_ubatch(int32_t count) {
    _n_ubatch = count;
}
int32_t GodotChorus::get_n_ubatch() const {
    return _n_ubatch;
}

void GodotChorus::set_main_gpu(int32_t index) {
    _main_gpu = index;
}
int32_t GodotChorus::get_main_gpu() const {
    return _main_gpu;
}

void GodotChorus::set_backend(BackendChoice backend) {
    if (is_loaded()) {
        UtilityFunctions::push_warning("[Chorus] backend changed while loaded; takes effect on the next load_model().");
    }
    _backend = backend;
}

GodotChorus::BackendChoice GodotChorus::get_backend() const {
    return _backend;
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

std::string GodotChorus::resolve_chat_template(std::string request_template) const {
    if (request_template.empty() && !_chat_template.is_empty())
        return std::string(_chat_template.utf8().get_data());
    return request_template;
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

    // --- BackendChoice enum ---
    BIND_ENUM_CONSTANT(BACKEND_LLAMA);
    BIND_ENUM_CONSTANT(BACKEND_ECHO);

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

    ClassDB::bind_method(D_METHOD("set_context_size", "size"), &GodotChorus::set_context_size);
    ClassDB::bind_method(D_METHOD("get_context_size"), &GodotChorus::get_context_size);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "context_size", PROPERTY_HINT_RANGE, "128,65536,128"),
        "set_context_size",
        "get_context_size"
    );

    ClassDB::bind_method(D_METHOD("set_thread_count", "count"), &GodotChorus::set_thread_count);
    ClassDB::bind_method(D_METHOD("get_thread_count"), &GodotChorus::get_thread_count);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "thread_count", PROPERTY_HINT_RANGE, "1,32,1"),
        "set_thread_count",
        "get_thread_count"
    );

    ClassDB::bind_method(D_METHOD("set_use_gpu", "use"), &GodotChorus::set_use_gpu);
    ClassDB::bind_method(D_METHOD("get_use_gpu"), &GodotChorus::get_use_gpu);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gpu"), "set_use_gpu", "get_use_gpu");

    ClassDB::bind_method(D_METHOD("set_gpu_layers", "layers"), &GodotChorus::set_gpu_layers);
    ClassDB::bind_method(D_METHOD("get_gpu_layers"), &GodotChorus::get_gpu_layers);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "gpu_layers", PROPERTY_HINT_RANGE, Chorus::GodotAdapter::GPU_LAYERS_PROPERTY_HINT),
        "set_gpu_layers",
        "get_gpu_layers"
    );

    ClassDB::bind_method(D_METHOD("set_num_slots", "count"), &GodotChorus::set_num_slots);
    ClassDB::bind_method(D_METHOD("get_num_slots"), &GodotChorus::get_num_slots);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "num_slots", PROPERTY_HINT_RANGE, "1,32,1"), "set_num_slots", "get_num_slots"
    );

    ClassDB::bind_method(D_METHOD("set_tokens_per_tick", "count"), &GodotChorus::set_tokens_per_tick);
    ClassDB::bind_method(D_METHOD("get_tokens_per_tick"), &GodotChorus::get_tokens_per_tick);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "tokens_per_tick", PROPERTY_HINT_RANGE, "1,4096,1"),
        "set_tokens_per_tick",
        "get_tokens_per_tick"
    );

    ClassDB::bind_method(D_METHOD("set_n_batch", "count"), &GodotChorus::set_n_batch);
    ClassDB::bind_method(D_METHOD("get_n_batch"), &GodotChorus::get_n_batch);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_batch", PROPERTY_HINT_RANGE, "1,65536,1"), "set_n_batch", "get_n_batch");

    ClassDB::bind_method(D_METHOD("set_n_ubatch", "count"), &GodotChorus::set_n_ubatch);
    ClassDB::bind_method(D_METHOD("get_n_ubatch"), &GodotChorus::get_n_ubatch);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "n_ubatch", PROPERTY_HINT_RANGE, "1,65536,1"), "set_n_ubatch", "get_n_ubatch"
    );

    ClassDB::bind_method(D_METHOD("set_main_gpu", "index"), &GodotChorus::set_main_gpu);
    ClassDB::bind_method(D_METHOD("get_main_gpu"), &GodotChorus::get_main_gpu);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "main_gpu", PROPERTY_HINT_RANGE, "0,15,1"), "set_main_gpu", "get_main_gpu");

    ClassDB::bind_method(D_METHOD("set_backend", "backend"), &GodotChorus::set_backend);
    ClassDB::bind_method(D_METHOD("get_backend"), &GodotChorus::get_backend);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "backend", PROPERTY_HINT_ENUM, "Llama,Echo"), "set_backend", "get_backend");

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
