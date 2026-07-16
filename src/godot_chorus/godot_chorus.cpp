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
        case Chorus::RuntimeEvent::Kind::Complete:
            emit_signal("generation_complete", event.request_id, session, String(event.text.c_str()));
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
        config.backend_options["llama"] = Chorus::OptionMap{
            {"context_size", int64_t{_context_size}},
            {"thread_count", int64_t{_thread_count}},
            {"use_gpu", _use_gpu},
            {"gpu_layers", int64_t{_gpu_layers}},
            {"num_slots", int64_t{_num_slots}},
            {"tokens_per_tick", int64_t{_tokens_per_tick}},
            {"n_batch", int64_t{_n_batch}},
            {"n_ubatch", int64_t{_n_ubatch}},
            {"main_gpu", int64_t{_main_gpu}},
        };
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

    auto result = _runtime.submit(std::get<Chorus::GenerationRequest>(normalized));
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
        "generation_complete",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING, "session"),
        PropertyInfo(Variant::STRING, "full_text")
    ));
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

    // --- Core methods ---
    ClassDB::bind_method(D_METHOD("load_model"), &GodotChorus::load_model);
    ClassDB::bind_method(D_METHOD("stop_all"), &GodotChorus::stop_all);
    ClassDB::bind_method(D_METHOD("is_loaded"), &GodotChorus::is_loaded);
    ClassDB::bind_method(D_METHOD("generate", "request"), &GodotChorus::generate);
    ClassDB::bind_method(D_METHOD("cancel_request", "request_id"), &GodotChorus::cancel_request);
    ClassDB::bind_method(D_METHOD("is_request_active", "request_id"), &GodotChorus::is_request_active);
    ClassDB::bind_method(D_METHOD("active_request_for_session", "session"), &GodotChorus::active_request_for_session);
    ClassDB::bind_method(D_METHOD("similarity_cos", "array1", "array2"), &GodotChorus::similarity_cos);

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
        PropertyInfo(Variant::INT, "gpu_layers", PROPERTY_HINT_RANGE, "0,999,1"), "set_gpu_layers", "get_gpu_layers"
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
}
