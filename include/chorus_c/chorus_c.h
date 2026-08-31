/*
 * C ABI host adapter over ChorusRuntime.
 *
 * Every function taking a chorus_runtime must be called from one host thread.
 * Input strings are borrowed UTF-8 NUL-terminated values and are copied before
 * return. Returned storage is owned as documented by each function.
 */

#ifndef CHORUS_C_CHORUS_C_H
#define CHORUS_C_CHORUS_C_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(CHORUS_C_BUILD)
#define CHORUS_API __declspec(dllexport)
#else
#define CHORUS_API __declspec(dllimport)
#endif
#else
#define CHORUS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chorus_runtime chorus_runtime;
typedef struct chorus_options chorus_options;
typedef struct chorus_request chorus_request;

typedef int64_t chorus_request_id;

typedef enum chorus_error {
    CHORUS_OK = 0,
    CHORUS_ERR_MODEL_LOAD = 1,
    CHORUS_ERR_CONTEXT_INIT = 2,
    CHORUS_ERR_DECODE = 3,
    CHORUS_ERR_TOKENIZE = 4,
    CHORUS_ERR_INVALID_REQUEST = 5,
    CHORUS_ERR_ENGINE_NOT_READY = 6,
    CHORUS_ERR_CANCELLED = 7,
    CHORUS_ERR_UNSUPPORTED_MODEL_FORMAT = 8,
    CHORUS_ERR_UNSUPPORTED_FEATURE = 9,
    CHORUS_ERR_UNSUPPORTED_OPTION = 10,
    CHORUS_ERR_SESSION_BUSY = 11,
    CHORUS_ERR_UNKNOWN = 12,
} chorus_error;

typedef enum chorus_provider {
    CHORUS_PROVIDER_LLAMA = 0,
    CHORUS_PROVIDER_ECHO = 1,
} chorus_provider;

typedef enum chorus_event_kind {
    CHORUS_EVENT_TOKEN = 0,
    CHORUS_EVENT_REASONING_TOKEN = 1,
    CHORUS_EVENT_COMPLETE = 2,
    CHORUS_EVENT_ERROR = 3,
    CHORUS_EVENT_HISTORY_TRUNCATED = 4,
    CHORUS_EVENT_ENGINE_FAILED = 5,
} chorus_event_kind;

typedef enum chorus_turn_outcome {
    CHORUS_TURN_NONE = 0,
    CHORUS_TURN_COMPLETED = 1,
    CHORUS_TURN_CANCELLED = 2,
    CHORUS_TURN_ERRORED = 3,
} chorus_turn_outcome;

typedef enum chorus_constraint_format {
    CHORUS_CONSTRAINT_GBNF = 0,
    CHORUS_CONSTRAINT_JSON_SCHEMA = 1,
    CHORUS_CONSTRAINT_REGEX = 2,
    CHORUS_CONSTRAINT_LARK = 3,
} chorus_constraint_format;

typedef enum chorus_log_level {
    CHORUS_LOG_DEBUG = 0,
    CHORUS_LOG_INFO = 1,
    CHORUS_LOG_WARN = 2,
    CHORUS_LOG_ERROR = 3,
    CHORUS_LOG_FATAL = 4,
    CHORUS_LOG_OFF = 5,
} chorus_log_level;

#define CHORUS_LOG_LEVEL_DEFAULT CHORUS_LOG_WARN

typedef enum chorus_log_field_type {
    CHORUS_FIELD_INT = 0,
    CHORUS_FIELD_FLOAT = 1,
    CHORUS_FIELD_BOOL = 2,
    CHORUS_FIELD_STRING = 3,
} chorus_log_field_type;

typedef struct chorus_chat_message {
    const char* role;
    const char* content;
} chorus_chat_message;

/* Runtime-owned pointers remain valid until the next chorus_poll or free. */
typedef struct chorus_event {
    chorus_event_kind kind;
    chorus_request_id request_id;
    const char* session;
    const char* text;
    chorus_error error;
    const char* reasoning;
    int32_t dropped;
} chorus_event;

typedef struct chorus_log_field {
    const char* key;
    chorus_log_field_type type;
    union {
        int64_t int_value;
        double float_value;
        bool bool_value;
        const char* string_value;
    } value;
} chorus_log_field;

/* Runtime-owned pointers remain valid until the next chorus_poll_logs or free. */
typedef struct chorus_log_record {
    chorus_log_level level;
    const char* message;
    const chorus_log_field* fields;
    size_t field_count;
    chorus_request_id request_id;
    const char* session;
    double produced_at;
} chorus_log_record;

CHORUS_API uint32_t chorus_abi_version(void);
CHORUS_API const char* chorus_error_name(chorus_error error);
CHORUS_API void chorus_string_free(char* str);

CHORUS_API chorus_runtime* chorus_runtime_new(void);
CHORUS_API void chorus_runtime_free(chorus_runtime* rt);
CHORUS_API const char* chorus_last_error_message(const chorus_runtime* rt);

CHORUS_API chorus_options* chorus_options_new(void);
CHORUS_API void chorus_options_free(chorus_options* opts);
CHORUS_API chorus_error chorus_options_set_int(chorus_options* opts, const char* key, int64_t value);
CHORUS_API chorus_error chorus_options_set_float(chorus_options* opts, const char* key, double value);
CHORUS_API chorus_error chorus_options_set_bool(chorus_options* opts, const char* key, bool value);
CHORUS_API chorus_error chorus_options_set_string(chorus_options* opts, const char* key, const char* value);

/* model_path may be NULL only for CHORUS_PROVIDER_ECHO. */
CHORUS_API chorus_error chorus_load(
    chorus_runtime* rt,
    chorus_provider provider,
    const char* model_path,
    const chorus_options* options,
    chorus_log_level min_log_level
);
CHORUS_API bool chorus_is_loaded(const chorus_runtime* rt);
CHORUS_API void chorus_stop_all(chorus_runtime* rt);

CHORUS_API chorus_request* chorus_request_new(void);
CHORUS_API void chorus_request_free(chorus_request* req);

CHORUS_API chorus_error chorus_request_set_prompt(chorus_request* req, const char* prompt);
/* NULL clears the session and selects stateless generation. */
CHORUS_API chorus_error chorus_request_set_session(chorus_request* req, const char* session);
CHORUS_API chorus_error chorus_request_set_priority(chorus_request* req, int32_t priority);
CHORUS_API chorus_error chorus_request_set_stream(chorus_request* req, bool stream);
CHORUS_API chorus_error chorus_request_set_max_tokens(chorus_request* req, int32_t max_tokens);
CHORUS_API chorus_error chorus_request_set_temperature(chorus_request* req, float temperature);
CHORUS_API chorus_error chorus_request_set_top_k(chorus_request* req, int32_t top_k);
CHORUS_API chorus_error chorus_request_set_top_p(chorus_request* req, float top_p);
CHORUS_API chorus_error chorus_request_set_seed(chorus_request* req, uint64_t seed);
CHORUS_API chorus_error chorus_request_set_frequency_penalty(chorus_request* req, float penalty);
CHORUS_API chorus_error chorus_request_set_presence_penalty(chorus_request* req, float penalty);
CHORUS_API chorus_error chorus_request_add_stop(chorus_request* req, const char* sequence);
CHORUS_API chorus_error
chorus_request_set_constraint(chorus_request* req, chorus_constraint_format format, const char* source);
CHORUS_API chorus_error chorus_request_set_show_thinking(chorus_request* req, bool show_thinking);
CHORUS_API chorus_error chorus_request_set_provider_option_float(
    chorus_request* req, const char* provider, const char* key, double value
);
CHORUS_API chorus_error chorus_request_set_provider_option_int(
    chorus_request* req, const char* provider, const char* key, int64_t value
);
CHORUS_API chorus_error chorus_request_set_provider_option_bool(
    chorus_request* req, const char* provider, const char* key, bool value
);
CHORUS_API chorus_error chorus_request_set_provider_option_string(
    chorus_request* req, const char* provider, const char* key, const char* value
);
CHORUS_API chorus_error
chorus_request_add_inject(chorus_request* req, chorus_chat_message message, int32_t depth);
CHORUS_API chorus_error chorus_request_set_chat_template(chorus_request* req, const char* chat_template);

CHORUS_API chorus_error
chorus_generate(chorus_runtime* rt, const chorus_request* req, chorus_request_id* out_request_id);
CHORUS_API chorus_error
chorus_regenerate(chorus_runtime* rt, const chorus_request* req, chorus_request_id* out_request_id);
CHORUS_API bool chorus_cancel(chorus_runtime* rt, chorus_request_id request_id);
CHORUS_API bool chorus_is_request_active(const chorus_runtime* rt, chorus_request_id request_id);
CHORUS_API chorus_request_id chorus_active_request_for_session(const chorus_runtime* rt, const char* session);

/* The returned pointer is non-NULL even when out_count receives zero. */
CHORUS_API const chorus_event* chorus_poll(chorus_runtime* rt, size_t* out_count);
/* The returned pointer is non-NULL even when out_count receives zero. */
CHORUS_API const chorus_log_record* chorus_poll_logs(chorus_runtime* rt, size_t* out_count);

CHORUS_API chorus_error chorus_history_import(
    chorus_runtime* rt, const char* session, const chorus_chat_message* history, size_t count
);
/* Caller-owned deep snapshot; free with chorus_chat_messages_free. */
CHORUS_API chorus_error chorus_history_export(
    const chorus_runtime* rt, const char* session, chorus_chat_message** out_messages, size_t* out_count
);
CHORUS_API void chorus_chat_messages_free(chorus_chat_message* messages, size_t count);
CHORUS_API chorus_error chorus_history_clear(chorus_runtime* rt, const char* session);
CHORUS_API chorus_error
chorus_history_edit_message(chorus_runtime* rt, const char* session, int64_t index, const char* content);
/* Caller-owned string array; free with chorus_string_list_free. */
CHORUS_API chorus_error
chorus_list_conversations(const chorus_runtime* rt, char*** out_sessions, size_t* out_count);
CHORUS_API void chorus_string_list_free(char** strings, size_t count);
CHORUS_API chorus_error chorus_reset_context(chorus_runtime* rt);
CHORUS_API chorus_turn_outcome chorus_last_turn_outcome(const chorus_runtime* rt, const char* session);
/* Caller owns the returned string and frees it with chorus_string_free. */
CHORUS_API char* chorus_render_prompt(const chorus_runtime* rt, const char* session, const chorus_request* req);

#ifdef __cplusplus
}
#endif

#endif
