#include "godot_chorus/godot_chorus.hpp"

#include <cmath>
#include <memory>

#include "chorus/engine_factory.hpp"

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
        switch (event.kind) {
        case Chorus::RuntimeEvent::Kind::Token:
            emit_signal("token_generated", event.request_id, String(event.text.c_str()));
            break;
        case Chorus::RuntimeEvent::Kind::Complete:
            emit_signal("generation_complete", event.request_id, String(event.text.c_str()));
            break;
        case Chorus::RuntimeEvent::Kind::Error:
            emit_signal("generation_error", event.request_id, to_godot(event.error), String(event.text.c_str()));
            break;
        }
    }
}

// ===========================================================================
// Core API
// ===========================================================================

bool GodotChorus::load_model() {
    if (_backend != BACKEND_ECHO && _chorus_config.model_path.empty()) {
        UtilityFunctions::push_error("[Chorus] model_path is not set.");
        return false;
    }

    _chorus_config.log_callback = [](Chorus::LogLevel level, const std::string& msg) {
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

    auto engine = Chorus::make_engine(_backend == BACKEND_ECHO ? Chorus::Backend::Echo : Chorus::Backend::Llama);
    auto err = _runtime.load_engine(std::move(engine), _chorus_config);
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
    if (!request.has("prompt")) {
        UtilityFunctions::push_error("[Chorus] generate() requires a 'prompt' key in the request dictionary.");
        return -1;
    }

    Chorus::GenerationRequest gen_request;
    gen_request.prompt = ((String)request["prompt"]).utf8().get_data();
    gen_request.stream = request.has("stream") ? (bool)request["stream"] : false;
    gen_request.priority = request.has("priority") ? (int)(int64_t)request["priority"] : 0;

    if (request.has("max_tokens"))
        gen_request.config.max_tokens = (int32_t)(int64_t)request["max_tokens"];
    if (request.has("temperature"))
        gen_request.config.temperature = (float)request["temperature"];
    if (request.has("top_k"))
        gen_request.config.top_k = (int32_t)(int64_t)request["top_k"];
    if (request.has("top_p"))
        gen_request.config.top_p = (float)request["top_p"];
    if (request.has("repeat_penalty"))
        gen_request.config.repeat_penalty = (float)request["repeat_penalty"];
    if (request.has("seed"))
        gen_request.config.seed = (uint32_t)(int64_t)request["seed"];
    if (request.has("grammar"))
        gen_request.config.grammar = ((String)request["grammar"]).utf8().get_data();

    auto result = _runtime.submit(gen_request);
    if (!result.ok()) {
        String message = String("[Chorus] generate() rejected: ") + chorus_error_name(result.error);
        if (result.error == Chorus::ChorusError::EngineNotReady)
            message += ". Call load_model() first.";
        UtilityFunctions::push_error(message);
        return -1;
    }
    return result.request_id;
}

// ===========================================================================
// Properties
// ===========================================================================

void GodotChorus::set_model_path(const String& path) {
    _chorus_config.model_path = path.utf8().get_data();
}
String GodotChorus::get_model_path() const {
    return String(_chorus_config.model_path.c_str());
}

void GodotChorus::set_context_size(int32_t size) {
    _chorus_config.context_size = size;
}
int32_t GodotChorus::get_context_size() const {
    return _chorus_config.context_size;
}

void GodotChorus::set_thread_count(int32_t count) {
    _chorus_config.thread_count = count;
}
int32_t GodotChorus::get_thread_count() const {
    return _chorus_config.thread_count;
}

void GodotChorus::set_use_gpu(bool use) {
    _chorus_config.use_gpu = use;
}
bool GodotChorus::get_use_gpu() const {
    return _chorus_config.use_gpu;
}

void GodotChorus::set_gpu_layers(int32_t layers) {
    _chorus_config.gpu_layers = layers;
}
int32_t GodotChorus::get_gpu_layers() const {
    return _chorus_config.gpu_layers;
}

void GodotChorus::set_num_slots(int32_t count) {
    _chorus_config.num_slots = count;
}
int32_t GodotChorus::get_num_slots() const {
    return _chorus_config.num_slots;
}

void GodotChorus::set_tokens_per_tick(int32_t count) {
    _chorus_config.tokens_per_tick = count;
}
int32_t GodotChorus::get_tokens_per_tick() const {
    return _chorus_config.tokens_per_tick;
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
    ADD_SIGNAL(
        MethodInfo("token_generated", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "token"))
    );
    ADD_SIGNAL(MethodInfo(
        "generation_complete", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "full_text")
    ));
    ADD_SIGNAL(MethodInfo(
        "generation_error",
        PropertyInfo(Variant::INT, "request_id"),
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
    BIND_ENUM_CONSTANT(ERR_UNKNOWN);

    // --- BackendChoice enum ---
    BIND_ENUM_CONSTANT(BACKEND_LLAMA);
    BIND_ENUM_CONSTANT(BACKEND_ECHO);

    // --- Core methods ---
    ClassDB::bind_method(D_METHOD("load_model"), &GodotChorus::load_model);
    ClassDB::bind_method(D_METHOD("stop_all"), &GodotChorus::stop_all);
    ClassDB::bind_method(D_METHOD("is_loaded"), &GodotChorus::is_loaded);
    ClassDB::bind_method(D_METHOD("generate", "request"), &GodotChorus::generate);
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

    ClassDB::bind_method(D_METHOD("set_backend", "backend"), &GodotChorus::set_backend);
    ClassDB::bind_method(D_METHOD("get_backend"), &GodotChorus::get_backend);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "backend", PROPERTY_HINT_ENUM, "Llama,Echo"), "set_backend", "get_backend");
}
