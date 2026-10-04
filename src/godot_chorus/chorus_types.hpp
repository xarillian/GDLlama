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
    static godot::Ref<ChorusInjectedMessage>
    create(ChorusRole::Value role, const godot::String& content, int64_t depth = 0);
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

    void set_stream(bool value);
    bool get_stream() const;
    void set_max_tokens(int64_t value);
    int64_t get_max_tokens() const;
    bool has_max_tokens() const;
    void clear_max_tokens();
    void set_temperature(double value);
    double get_temperature() const;
    bool has_temperature() const;
    void clear_temperature();
    void set_top_k(int64_t value);
    int64_t get_top_k() const;
    bool has_top_k() const;
    void clear_top_k();
    void set_top_p(double value);
    double get_top_p() const;
    bool has_top_p() const;
    void clear_top_p();
    void set_seed(int64_t value);
    int64_t get_seed() const;
    bool has_seed() const;
    void clear_seed();
    void set_frequency_penalty(double value);
    double get_frequency_penalty() const;
    bool has_frequency_penalty() const;
    void clear_frequency_penalty();
    void set_presence_penalty(double value);
    double get_presence_penalty() const;
    bool has_presence_penalty() const;
    void clear_presence_penalty();
    void set_stop(godot::PackedStringArray value);
    godot::PackedStringArray get_stop() const;
    bool has_stop() const;
    void clear_stop();
    void set_show_thinking(bool value);
    bool get_show_thinking() const;
    bool has_show_thinking() const;
    void clear_show_thinking();
    void set_constraint(ChorusConstraintFormat::Value format, const godot::String& source);
    void set_unconstrained();
    bool has_constraint() const;
    bool is_unconstrained() const;
    void clear_constraint();
    ChorusConstraintFormat::Value get_constraint_format() const;
    godot::String get_constraint_source() const;
    void set_provider_options(const godot::Dictionary& value);
    godot::Dictionary get_provider_options() const;
    void clear_provider_options();
    void clear_provider_option(const godot::String& provider, const godot::String& key);
    void set_inject(const godot::TypedArray<ChorusInjectedMessage>& value);
    godot::TypedArray<ChorusInjectedMessage> get_inject() const;
    void set_chat_template(const godot::String& value);
    godot::String get_chat_template() const;
    bool has_chat_template() const;
    void clear_chat_template();
    bool requires_nonempty_session() const;
    bool _set(const godot::StringName& property, const godot::Variant& value);
    bool _get(const godot::StringName& property, godot::Variant& value) const;
    void _get_property_list(godot::List<godot::PropertyInfo>* list) const;

  private:
    bool _stream = false;
    bool _has_max_tokens = false;
    int64_t _max_tokens = 0;
    bool _has_temperature = false;
    double _temperature = 0.0;
    bool _has_top_k = false;
    int64_t _top_k = 0;
    bool _has_top_p = false;
    double _top_p = 0.0;
    bool _has_seed = false;
    int64_t _seed = 0;
    bool _has_frequency_penalty = false;
    double _frequency_penalty = 0.0;
    bool _has_presence_penalty = false;
    double _presence_penalty = 0.0;
    bool _has_stop = false;
    godot::PackedStringArray _stop;
    bool _has_constraint = false;
    bool _unconstrained = false;
    ChorusConstraintFormat::Value _constraint_format = ChorusConstraintFormat::GBNF;
    godot::String _constraint_source;
    bool _has_show_thinking = false;
    bool _show_thinking = false;
    godot::Dictionary _provider_options;
    godot::TypedArray<ChorusInjectedMessage> _inject;
    bool _has_chat_template = false;
    godot::String _chat_template;
    bool _chat_session_required = false;
};

class ChorusEmbeddingRequest : public ChorusInferenceRequest {
    GDCLASS(ChorusEmbeddingRequest, ChorusInferenceRequest);

  protected:
    static void _bind_methods();

  public:
    static godot::Ref<ChorusEmbeddingRequest>
    create(const godot::String& content, const godot::StringName& session = godot::StringName());
};

class ChorusLoadPhase : public godot::RefCounted {
    GDCLASS(ChorusLoadPhase, godot::RefCounted);

  protected:
    static void _bind_methods();

  public:
    enum Value { RELEASING_ENGINE, LOADING_MODEL, INITIALIZING_ENGINE };
};

class ChorusLoadResult : public godot::RefCounted {
    GDCLASS(ChorusLoadResult, godot::RefCounted);

  protected:
    static void _bind_methods();

  public:
    int64_t get_load_id() const;
    int get_error() const;
    godot::String get_message() const;
    bool get_accepted() const;
    void set_result(int64_t load_id, int error, const godot::String& message);

  private:
    int64_t _load_id = -1;
    int _error = 5;
    godot::String _message;
};

class ChorusSubmitResult : public godot::RefCounted {
    GDCLASS(ChorusSubmitResult, godot::RefCounted);

  protected:
    static void _bind_methods();

  public:
    ChorusSubmitResult() = default;
    ChorusSubmitResult(
        const godot::Ref<ChorusInferenceRequest>& request,
        int64_t request_id,
        int64_t request_message_id,
        int64_t response_message_id,
        int error,
        const godot::String& message
    );
    godot::Ref<ChorusInferenceRequest> get_request() const;
    int64_t get_request_id() const;
    int64_t get_request_message_id() const;
    int64_t get_response_message_id() const;
    int get_error() const;
    godot::String get_message() const;
    bool get_accepted() const;

  private:
    godot::Ref<ChorusInferenceRequest> _request;
    int64_t _request_id = -1;
    int64_t _request_message_id = -1;
    int64_t _response_message_id = -1;
    int _error = 0;
    godot::String _message;
};

class ChorusResult : public godot::RefCounted {
    GDCLASS(ChorusResult, godot::RefCounted);

  protected:
    static void _bind_methods();

  public:
    ChorusResult() = default;
    ChorusResult(int error, const godot::String& message);
    int get_error() const;
    godot::String get_message() const;
    bool get_ok() const;

  private:
    int _error = 0;
    godot::String _message;
};

class ChorusGenerationUsage : public godot::RefCounted {
    GDCLASS(ChorusGenerationUsage, godot::RefCounted);

  protected:
    static void _bind_methods();

  public:
    ChorusGenerationUsage() = default;
    ChorusGenerationUsage(int64_t prompt_tokens, int64_t cached_prompt_tokens, int64_t generated_tokens);
    int64_t get_prompt_tokens() const;
    int64_t get_cached_prompt_tokens() const;
    int64_t get_generated_tokens() const;

  private:
    int64_t _prompt_tokens = 0;
    int64_t _cached_prompt_tokens = 0;
    int64_t _generated_tokens = 0;
};

VARIANT_ENUM_CAST(ChorusLoadPhase::Value);
VARIANT_ENUM_CAST(ChorusRole::Value);
VARIANT_ENUM_CAST(ChorusExecution::Value);
VARIANT_ENUM_CAST(ChorusConstraintFormat::Value);
