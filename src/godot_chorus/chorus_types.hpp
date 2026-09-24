#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>

class ChorusRole : public godot::RefCounted {
    GDCLASS(ChorusRole, godot::RefCounted);
  protected:
    static void _bind_methods();
  public:
    enum Value { SYSTEM, USER, ASSISTANT };
};

class ChorusOverrideState : public godot::RefCounted {
    GDCLASS(ChorusOverrideState, godot::RefCounted);
  protected:
    static void _bind_methods();
  public:
    enum Value { INHERIT, SET, CLEAR };
};

class ChorusExecution : public godot::RefCounted {
    GDCLASS(ChorusExecution, godot::RefCounted);
  protected:
    static void _bind_methods();
  public:
    enum Value { SHARED, EXCLUSIVE };
};

class ChorusConstraintFormat : public godot::RefCounted {
    GDCLASS(ChorusConstraintFormat, godot::RefCounted);
  protected:
    static void _bind_methods();
  public:
    enum Value { GBNF, JSON_SCHEMA, REGEX, LARK };
};

class ChorusInferenceRequest : public godot::Resource {
    GDCLASS(ChorusInferenceRequest, godot::Resource);
  protected:
    static void _bind_methods();
  public:
    void set_content(const godot::String& value);
    godot::String get_content() const;
    void set_session(const godot::StringName& value);
    godot::StringName get_session() const;
    void set_priority(int64_t value);
    int64_t get_priority() const;
    void set_execution(ChorusExecution::Value value);
    ChorusExecution::Value get_execution() const;
  protected:
    godot::String _content;
    godot::StringName _session;
    int64_t _priority = 0;
    ChorusExecution::Value _execution = ChorusExecution::SHARED;
};

class ChorusInjectedMessage : public godot::Resource {
    GDCLASS(ChorusInjectedMessage, godot::Resource);
  protected:
    static void _bind_methods();
  public:
    static godot::Ref<ChorusInjectedMessage> create(ChorusRole::Value role, const godot::String& content, int64_t depth = 0);
    void set_role(ChorusRole::Value value);
    ChorusRole::Value get_role() const;
    void set_content(const godot::String& value);
    godot::String get_content() const;
    void set_depth(int64_t value);
    int64_t get_depth() const;
  private:
    ChorusRole::Value _role = ChorusRole::USER;
    godot::String _content;
    int64_t _depth = 0;
};

class ChorusMessage : public godot::Resource {
    GDCLASS(ChorusMessage, godot::Resource);
  protected:
    static void _bind_methods();
  public:
    static godot::Ref<ChorusMessage> create(int64_t id, ChorusRole::Value role, const godot::String& content);
    void set_id(int64_t value);
    int64_t get_id() const;
    void set_role(ChorusRole::Value value);
    ChorusRole::Value get_role() const;
    void set_content(const godot::String& value);
    godot::String get_content() const;
  private:
    int64_t _id = -1;
    ChorusRole::Value _role = ChorusRole::USER;
    godot::String _content;
};

class ChorusRequest : public ChorusInferenceRequest {
    GDCLASS(ChorusRequest, ChorusInferenceRequest);
  protected:
    static void _bind_methods();
  public:
    static godot::Ref<ChorusRequest> chat(const godot::StringName& session, const godot::String& content);
    static godot::Ref<ChorusRequest> stateless(const godot::String& content);
    static godot::Ref<ChorusRequest> regeneration(const godot::StringName& session);

    void set_stream(bool value); bool get_stream() const;
    void set_max_tokens_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_max_tokens_state() const;
    void set_max_tokens_value(int64_t value); int64_t get_max_tokens() const;
    void set_temperature_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_temperature_state() const;
    void set_temperature_value(double value); double get_temperature() const;
    void set_top_k_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_top_k_state() const;
    void set_top_k_value(int64_t value); int64_t get_top_k() const;
    void set_top_p_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_top_p_state() const;
    void set_top_p_value(double value); double get_top_p() const;
    void set_seed_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_seed_state() const;
    void set_seed_value(int64_t value); int64_t get_seed() const;
    void set_frequency_penalty_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_frequency_penalty_state() const;
    void set_frequency_penalty_value(double value); double get_frequency_penalty() const;
    void set_presence_penalty_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_presence_penalty_state() const;
    void set_presence_penalty_value(double value); double get_presence_penalty() const;
    void set_stop_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_stop_state() const;
    void set_stop_value(godot::PackedStringArray value); godot::PackedStringArray get_stop() const;
    void set_constraint_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_constraint_state() const;
    void set_constraint_format(ChorusConstraintFormat::Value value); ChorusConstraintFormat::Value get_constraint_format() const;
    void set_constraint_source(const godot::String& value); godot::String get_constraint_source() const;
    void set_show_thinking_state(ChorusOverrideState::Value value); ChorusOverrideState::Value get_show_thinking_state() const;
    void set_show_thinking_value(bool value); bool get_show_thinking() const;
    void set_provider_options(const godot::Dictionary& value); godot::Dictionary get_provider_options() const;
    void set_provider_option_erasures(const godot::PackedStringArray& value); godot::PackedStringArray get_provider_option_erasures() const;
    void set_inject(const godot::TypedArray<ChorusInjectedMessage>& value); godot::TypedArray<ChorusInjectedMessage> get_inject() const;
    void set_chat_template(const godot::String& value); godot::String get_chat_template() const;

    void set_max_tokens(int64_t value); void clear_max_tokens(); void inherit_max_tokens();
    void set_temperature(double value); void clear_temperature(); void inherit_temperature();
    void set_top_k(int64_t value); void clear_top_k(); void inherit_top_k();
    void set_top_p(double value); void clear_top_p(); void inherit_top_p();
    void set_seed(int64_t value); void clear_seed(); void inherit_seed();
    void set_frequency_penalty(double value); void clear_frequency_penalty(); void inherit_frequency_penalty();
    void set_presence_penalty(double value); void clear_presence_penalty(); void inherit_presence_penalty();
    void set_stop(godot::PackedStringArray value); void clear_stop(); void inherit_stop();
    void set_constraint(ChorusConstraintFormat::Value format, const godot::String& source); void clear_constraint(); void inherit_constraint();
    void set_show_thinking(bool value); void clear_show_thinking(); void inherit_show_thinking();
    bool requires_nonempty_session() const;
  private:
    bool _stream = false;
    ChorusOverrideState::Value _max_tokens_state = ChorusOverrideState::INHERIT; int64_t _max_tokens = 0;
    ChorusOverrideState::Value _temperature_state = ChorusOverrideState::INHERIT; double _temperature = 0.0;
    ChorusOverrideState::Value _top_k_state = ChorusOverrideState::INHERIT; int64_t _top_k = 0;
    ChorusOverrideState::Value _top_p_state = ChorusOverrideState::INHERIT; double _top_p = 0.0;
    ChorusOverrideState::Value _seed_state = ChorusOverrideState::INHERIT; int64_t _seed = 0;
    ChorusOverrideState::Value _frequency_penalty_state = ChorusOverrideState::INHERIT; double _frequency_penalty = 0.0;
    ChorusOverrideState::Value _presence_penalty_state = ChorusOverrideState::INHERIT; double _presence_penalty = 0.0;
    ChorusOverrideState::Value _stop_state = ChorusOverrideState::INHERIT; godot::PackedStringArray _stop;
    ChorusOverrideState::Value _constraint_state = ChorusOverrideState::INHERIT; ChorusConstraintFormat::Value _constraint_format = ChorusConstraintFormat::GBNF; godot::String _constraint_source;
    ChorusOverrideState::Value _show_thinking_state = ChorusOverrideState::INHERIT; bool _show_thinking = false;
    godot::Dictionary _provider_options;
    godot::PackedStringArray _provider_option_erasures;
    godot::TypedArray<ChorusInjectedMessage> _inject;
    godot::String _chat_template;
    bool _chat_session_required = false;
};

class ChorusEmbeddingRequest : public ChorusInferenceRequest {
    GDCLASS(ChorusEmbeddingRequest, ChorusInferenceRequest);
  protected:
    static void _bind_methods();
  public:
    static godot::Ref<ChorusEmbeddingRequest> create(const godot::String& content, const godot::StringName& session = godot::StringName());
};

class ChorusSubmitResult : public godot::RefCounted {
    GDCLASS(ChorusSubmitResult, godot::RefCounted);
  protected:
    static void _bind_methods();
  public:
    ChorusSubmitResult() = default;
    ChorusSubmitResult(const godot::Ref<ChorusInferenceRequest>& request, int64_t request_id, int64_t request_message_id, int64_t response_message_id, int error, const godot::String& message);
    godot::Ref<ChorusInferenceRequest> get_request() const; int64_t get_request_id() const; int64_t get_request_message_id() const; int64_t get_response_message_id() const; int get_error() const; godot::String get_message() const; bool get_accepted() const;
  private:
    godot::Ref<ChorusInferenceRequest> _request; int64_t _request_id = -1; int64_t _request_message_id = -1; int64_t _response_message_id = -1; int _error = 0; godot::String _message;
};

class ChorusResult : public godot::RefCounted {
    GDCLASS(ChorusResult, godot::RefCounted);
  protected:
    static void _bind_methods();
  public:
    ChorusResult() = default; ChorusResult(int error, const godot::String& message);
    int get_error() const; godot::String get_message() const; bool get_ok() const;
  private: int _error = 0; godot::String _message;
};

VARIANT_ENUM_CAST(ChorusRole::Value);
VARIANT_ENUM_CAST(ChorusOverrideState::Value);
VARIANT_ENUM_CAST(ChorusExecution::Value);
VARIANT_ENUM_CAST(ChorusConstraintFormat::Value);
