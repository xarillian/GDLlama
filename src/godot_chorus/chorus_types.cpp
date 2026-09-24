#include "godot_chorus/chorus_types.hpp"

using namespace godot;

#define BIND_NAMESPACE(TYPE, ...) \
void TYPE::_bind_methods() { __VA_ARGS__ }
#define BIND_VALUE(TYPE, NAME, SETTER, GETTER, VARIANT_TYPE) \
    ClassDB::bind_method(D_METHOD(#SETTER, "value"), &TYPE::SETTER); \
    ClassDB::bind_method(D_METHOD(#GETTER), &TYPE::GETTER); \
    ADD_PROPERTY(PropertyInfo(VARIANT_TYPE, NAME), #SETTER, #GETTER)
#define BIND_STATE(TYPE, NAME, SETTER, GETTER) \
    ClassDB::bind_method(D_METHOD(#SETTER, "value"), &TYPE::SETTER); \
    ClassDB::bind_method(D_METHOD(#GETTER), &TYPE::GETTER); \
    ADD_PROPERTY(PropertyInfo(Variant::INT, NAME, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusOverrideState.Value"), #SETTER, #GETTER)

BIND_NAMESPACE(ChorusRole,
    BIND_ENUM_CONSTANT(SYSTEM); BIND_ENUM_CONSTANT(USER); BIND_ENUM_CONSTANT(ASSISTANT);
)
BIND_NAMESPACE(ChorusOverrideState,
    BIND_ENUM_CONSTANT(INHERIT); BIND_ENUM_CONSTANT(SET); BIND_ENUM_CONSTANT(CLEAR);
)
BIND_NAMESPACE(ChorusExecution,
    BIND_ENUM_CONSTANT(SHARED); BIND_ENUM_CONSTANT(EXCLUSIVE);
)
BIND_NAMESPACE(ChorusConstraintFormat,
    BIND_ENUM_CONSTANT(GBNF); BIND_ENUM_CONSTANT(JSON_SCHEMA); BIND_ENUM_CONSTANT(REGEX); BIND_ENUM_CONSTANT(LARK);
)

void ChorusInferenceRequest::set_content(const String& value) { _content = value; }
String ChorusInferenceRequest::get_content() const { return _content; }
void ChorusInferenceRequest::set_session(const StringName& value) { _session = value; }
StringName ChorusInferenceRequest::get_session() const { return _session; }
void ChorusInferenceRequest::set_priority(int64_t value) { _priority = value; }
int64_t ChorusInferenceRequest::get_priority() const { return _priority; }
void ChorusInferenceRequest::set_execution(ChorusExecution::Value value) { _execution = value; }
ChorusExecution::Value ChorusInferenceRequest::get_execution() const { return _execution; }
void ChorusInferenceRequest::_bind_methods() {
    BIND_VALUE(ChorusInferenceRequest, "content", set_content, get_content, Variant::STRING);
    BIND_VALUE(ChorusInferenceRequest, "session", set_session, get_session, Variant::STRING_NAME);
    BIND_VALUE(ChorusInferenceRequest, "priority", set_priority, get_priority, Variant::INT);
    ClassDB::bind_method(D_METHOD("set_execution", "value"), &ChorusInferenceRequest::set_execution);
    ClassDB::bind_method(D_METHOD("get_execution"), &ChorusInferenceRequest::get_execution);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "execution", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusExecution.Value"), "set_execution", "get_execution");
}

Ref<ChorusInjectedMessage> ChorusInjectedMessage::create(ChorusRole::Value role, const String& content, int64_t depth) { Ref<ChorusInjectedMessage> value; value.instantiate(); value->_role = role; value->_content = content; value->_depth = depth; return value; }
void ChorusInjectedMessage::set_role(ChorusRole::Value value) { _role = value; }
ChorusRole::Value ChorusInjectedMessage::get_role() const { return _role; }
void ChorusInjectedMessage::set_content(const String& value) { _content = value; }
String ChorusInjectedMessage::get_content() const { return _content; }
void ChorusInjectedMessage::set_depth(int64_t value) { _depth = value; }
int64_t ChorusInjectedMessage::get_depth() const { return _depth; }
void ChorusInjectedMessage::_bind_methods() {
    ClassDB::bind_static_method("ChorusInjectedMessage", D_METHOD("create", "role", "content", "depth"), &ChorusInjectedMessage::create, DEFVAL(0));
    ClassDB::bind_method(D_METHOD("set_role", "value"), &ChorusInjectedMessage::set_role); ClassDB::bind_method(D_METHOD("get_role"), &ChorusInjectedMessage::get_role);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "role", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusRole.Value"), "set_role", "get_role");
    BIND_VALUE(ChorusInjectedMessage, "content", set_content, get_content, Variant::STRING);
    BIND_VALUE(ChorusInjectedMessage, "depth", set_depth, get_depth, Variant::INT);
}

Ref<ChorusMessage> ChorusMessage::create(int64_t id, ChorusRole::Value role, const String& content) { Ref<ChorusMessage> value; value.instantiate(); value->_id = id; value->_role = role; value->_content = content; return value; }
void ChorusMessage::set_id(int64_t value) { _id = value; }
int64_t ChorusMessage::get_id() const { return _id; }
void ChorusMessage::set_role(ChorusRole::Value value) { _role = value; }
ChorusRole::Value ChorusMessage::get_role() const { return _role; }
void ChorusMessage::set_content(const String& value) { _content = value; }
String ChorusMessage::get_content() const { return _content; }
void ChorusMessage::_bind_methods() {
    ClassDB::bind_static_method("ChorusMessage", D_METHOD("create", "id", "role", "content"), &ChorusMessage::create);
    BIND_VALUE(ChorusMessage, "id", set_id, get_id, Variant::INT);
    ClassDB::bind_method(D_METHOD("set_role", "value"), &ChorusMessage::set_role); ClassDB::bind_method(D_METHOD("get_role"), &ChorusMessage::get_role);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "role", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusRole.Value"), "set_role", "get_role");
    BIND_VALUE(ChorusMessage, "content", set_content, get_content, Variant::STRING);
}

Ref<ChorusRequest> ChorusRequest::chat(const StringName& session, const String& content) { Ref<ChorusRequest> value; value.instantiate(); value->_session = session; value->_content = content; value->_chat_session_required = true; return value; }
Ref<ChorusRequest> ChorusRequest::stateless(const String& content) { Ref<ChorusRequest> value; value.instantiate(); value->_content = content; return value; }
Ref<ChorusRequest> ChorusRequest::regeneration(const StringName& session) { Ref<ChorusRequest> value; value.instantiate(); value->_session = session; return value; }
#define IMPLEMENT_FIELD(NAME, TYPE) \
void ChorusRequest::set_##NAME##_state(ChorusOverrideState::Value value) { _##NAME##_state = value; } \
ChorusOverrideState::Value ChorusRequest::get_##NAME##_state() const { return _##NAME##_state; } \
void ChorusRequest::set_##NAME##_value(TYPE value) { _##NAME = value; } \
TYPE ChorusRequest::get_##NAME() const { return _##NAME; } \
void ChorusRequest::set_##NAME(TYPE value) { _##NAME = value; _##NAME##_state = ChorusOverrideState::SET; } \
void ChorusRequest::clear_##NAME() { _##NAME##_state = ChorusOverrideState::CLEAR; } \
void ChorusRequest::inherit_##NAME() { _##NAME##_state = ChorusOverrideState::INHERIT; }
IMPLEMENT_FIELD(max_tokens, int64_t)
IMPLEMENT_FIELD(temperature, double)
IMPLEMENT_FIELD(top_k, int64_t)
IMPLEMENT_FIELD(top_p, double)
IMPLEMENT_FIELD(seed, int64_t)
IMPLEMENT_FIELD(frequency_penalty, double)
IMPLEMENT_FIELD(presence_penalty, double)
IMPLEMENT_FIELD(stop, PackedStringArray)
IMPLEMENT_FIELD(show_thinking, bool)
#undef IMPLEMENT_FIELD
void ChorusRequest::set_stream(bool value) { _stream = value; } bool ChorusRequest::get_stream() const { return _stream; }
void ChorusRequest::set_constraint_state(ChorusOverrideState::Value value) { _constraint_state = value; }
ChorusOverrideState::Value ChorusRequest::get_constraint_state() const { return _constraint_state; }
void ChorusRequest::set_constraint_format(ChorusConstraintFormat::Value value) { _constraint_format = value; }
ChorusConstraintFormat::Value ChorusRequest::get_constraint_format() const { return _constraint_format; }
void ChorusRequest::set_constraint_source(const String& value) { _constraint_source = value; }
String ChorusRequest::get_constraint_source() const { return _constraint_source; }
void ChorusRequest::set_constraint(ChorusConstraintFormat::Value format, const String& source) { _constraint_format = format; _constraint_source = source; _constraint_state = ChorusOverrideState::SET; }
void ChorusRequest::clear_constraint() { _constraint_state = ChorusOverrideState::CLEAR; }
void ChorusRequest::inherit_constraint() { _constraint_state = ChorusOverrideState::INHERIT; }
void ChorusRequest::set_provider_options(const Dictionary& value) { _provider_options = value; } Dictionary ChorusRequest::get_provider_options() const { return _provider_options; }
void ChorusRequest::set_provider_option_erasures(const PackedStringArray& value) { _provider_option_erasures = value; } PackedStringArray ChorusRequest::get_provider_option_erasures() const { return _provider_option_erasures; }
void ChorusRequest::set_inject(const TypedArray<ChorusInjectedMessage>& value) { _inject = value; } TypedArray<ChorusInjectedMessage> ChorusRequest::get_inject() const { return _inject; }
bool ChorusRequest::requires_nonempty_session() const { return _chat_session_required; }
void ChorusRequest::set_chat_template(const String& value) { _chat_template = value; } String ChorusRequest::get_chat_template() const { return _chat_template; }
#define BIND_OPTION(NAME, VARIANT_TYPE) \
    BIND_STATE(ChorusRequest, #NAME "_state", set_##NAME##_state, get_##NAME##_state); \
    ClassDB::bind_method(D_METHOD("set_" #NAME "_value", "value"), &ChorusRequest::set_##NAME##_value); ClassDB::bind_method(D_METHOD("get_" #NAME), &ChorusRequest::get_##NAME); \
    ADD_PROPERTY(PropertyInfo(VARIANT_TYPE, #NAME), "set_" #NAME "_value", "get_" #NAME); \
    ClassDB::bind_method(D_METHOD("set_" #NAME, "value"), &ChorusRequest::set_##NAME); ClassDB::bind_method(D_METHOD("clear_" #NAME), &ChorusRequest::clear_##NAME); ClassDB::bind_method(D_METHOD("inherit_" #NAME), &ChorusRequest::inherit_##NAME)
void ChorusRequest::_bind_methods() {
    ClassDB::bind_static_method("ChorusRequest", D_METHOD("chat", "session", "content"), &ChorusRequest::chat);
    ClassDB::bind_static_method("ChorusRequest", D_METHOD("stateless", "content"), &ChorusRequest::stateless);
    ClassDB::bind_static_method("ChorusRequest", D_METHOD("regeneration", "session"), &ChorusRequest::regeneration);
    BIND_VALUE(ChorusRequest, "stream", set_stream, get_stream, Variant::BOOL);
    BIND_OPTION(max_tokens, Variant::INT); BIND_OPTION(temperature, Variant::FLOAT); BIND_OPTION(top_k, Variant::INT); BIND_OPTION(top_p, Variant::FLOAT); BIND_OPTION(seed, Variant::INT); BIND_OPTION(frequency_penalty, Variant::FLOAT); BIND_OPTION(presence_penalty, Variant::FLOAT); BIND_OPTION(stop, Variant::PACKED_STRING_ARRAY); BIND_OPTION(show_thinking, Variant::BOOL);
    BIND_STATE(ChorusRequest, "constraint_state", set_constraint_state, get_constraint_state);
    ClassDB::bind_method(D_METHOD("set_constraint_format", "value"), &ChorusRequest::set_constraint_format); ClassDB::bind_method(D_METHOD("get_constraint_format"), &ChorusRequest::get_constraint_format);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "constraint_format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "ChorusConstraintFormat.Value"), "set_constraint_format", "get_constraint_format");
    BIND_VALUE(ChorusRequest, "constraint_source", set_constraint_source, get_constraint_source, Variant::STRING);
    ClassDB::bind_method(D_METHOD("set_constraint", "format", "source"), &ChorusRequest::set_constraint); ClassDB::bind_method(D_METHOD("clear_constraint"), &ChorusRequest::clear_constraint); ClassDB::bind_method(D_METHOD("inherit_constraint"), &ChorusRequest::inherit_constraint);
    BIND_VALUE(ChorusRequest, "provider_options", set_provider_options, get_provider_options, Variant::DICTIONARY);
    BIND_VALUE(ChorusRequest, "provider_option_erasures", set_provider_option_erasures, get_provider_option_erasures, Variant::PACKED_STRING_ARRAY);
    ClassDB::bind_method(D_METHOD("set_inject", "value"), &ChorusRequest::set_inject);
    ClassDB::bind_method(D_METHOD("get_inject"), &ChorusRequest::get_inject);
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "inject", PROPERTY_HINT_ARRAY_TYPE, "ChorusInjectedMessage"), "set_inject", "get_inject");
    BIND_VALUE(ChorusRequest, "chat_template", set_chat_template, get_chat_template, Variant::STRING);
}
#undef BIND_OPTION

Ref<ChorusEmbeddingRequest> ChorusEmbeddingRequest::create(const String& content, const StringName& session) { Ref<ChorusEmbeddingRequest> value; value.instantiate(); value->_content = content; value->_session = session; return value; }
void ChorusEmbeddingRequest::_bind_methods() { ClassDB::bind_static_method("ChorusEmbeddingRequest", D_METHOD("create", "content", "session"), &ChorusEmbeddingRequest::create, DEFVAL(StringName())); }

ChorusSubmitResult::ChorusSubmitResult(const Ref<ChorusInferenceRequest>& request, int64_t request_id, int64_t request_message_id, int64_t response_message_id, int error, const String& message) : _request(request), _request_id(request_id), _request_message_id(request_message_id), _response_message_id(response_message_id), _error(error), _message(message) {}
Ref<ChorusInferenceRequest> ChorusSubmitResult::get_request() const { return _request; } int64_t ChorusSubmitResult::get_request_id() const { return _request_id; } int64_t ChorusSubmitResult::get_request_message_id() const { return _request_message_id; } int64_t ChorusSubmitResult::get_response_message_id() const { return _response_message_id; } int ChorusSubmitResult::get_error() const { return _error; } String ChorusSubmitResult::get_message() const { return _message; } bool ChorusSubmitResult::get_accepted() const { return _error == 0; }
void ChorusSubmitResult::_bind_methods() { ClassDB::bind_method(D_METHOD("get_request"), &ChorusSubmitResult::get_request); ClassDB::bind_method(D_METHOD("get_request_id"), &ChorusSubmitResult::get_request_id); ClassDB::bind_method(D_METHOD("get_request_message_id"), &ChorusSubmitResult::get_request_message_id); ClassDB::bind_method(D_METHOD("get_response_message_id"), &ChorusSubmitResult::get_response_message_id); ClassDB::bind_method(D_METHOD("get_error"), &ChorusSubmitResult::get_error); ClassDB::bind_method(D_METHOD("get_message"), &ChorusSubmitResult::get_message); ClassDB::bind_method(D_METHOD("get_accepted"), &ChorusSubmitResult::get_accepted); ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "request", PROPERTY_HINT_RESOURCE_TYPE, "ChorusInferenceRequest", PROPERTY_USAGE_READ_ONLY), "", "get_request"); ADD_PROPERTY(PropertyInfo(Variant::INT, "request_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_request_id"); ADD_PROPERTY(PropertyInfo(Variant::INT, "request_message_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_request_message_id"); ADD_PROPERTY(PropertyInfo(Variant::INT, "response_message_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_response_message_id"); ADD_PROPERTY(PropertyInfo(Variant::INT, "error", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY, "GodotChorus.ErrorCode"), "", "get_error"); ADD_PROPERTY(PropertyInfo(Variant::STRING, "message", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_message"); ADD_PROPERTY(PropertyInfo(Variant::BOOL, "accepted", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_accepted"); }
ChorusResult::ChorusResult(int error, const String& message) : _error(error), _message(message) {} int ChorusResult::get_error() const { return _error; } String ChorusResult::get_message() const { return _message; } bool ChorusResult::get_ok() const { return _error == 0; }
void ChorusResult::_bind_methods() { ClassDB::bind_method(D_METHOD("get_error"), &ChorusResult::get_error); ClassDB::bind_method(D_METHOD("get_message"), &ChorusResult::get_message); ClassDB::bind_method(D_METHOD("get_ok"), &ChorusResult::get_ok); ADD_PROPERTY(PropertyInfo(Variant::INT, "error", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY, "GodotChorus.ErrorCode"), "", "get_error"); ADD_PROPERTY(PropertyInfo(Variant::STRING, "message", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_message"); ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ok", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_ok"); }
