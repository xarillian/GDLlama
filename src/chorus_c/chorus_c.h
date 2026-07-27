/* chorus_c.h -- C ABI host adapter over ChorusRuntime.
 *
 * A host adapter like src/godot_chorus: pure marshalling over the public API
 * (include/chorus/runtime/runtime.hpp). Language bindings (C#, Node, ...)
 * wrap this header; they never reach past it.
 *
 * Conventions:
 *   - Threading: every function taking a chorus_runtime* must be called from
 *     one thread, matching the runtime's host-thread confinement. No function
 *     here is thread-safe.
 *   - Strings: all char* parameters are borrowed, UTF-8, NUL-terminated, and
 *     copied before return. Returned strings and arrays are owned as
 *     documented per function; free with the matching chorus_..._free.
 *   - Optionals: builder setters left uncalled mean "backend default",
 *     mirroring the C++ std::optional fields.
 *   - Every fallible call returns chorus_error; CHORUS_OK is 0. A
 *     human-readable detail for the most recent failure on a runtime is
 *     available via chorus_last_error_message.
 *
 * Scaffold: declarations only. No implementation exists yet.
 */

#ifndef CHORUS_C_H
#define CHORUS_C_H

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

/* ========================================================================
 * Handles
 * ======================================================================== */

typedef struct chorus_runtime chorus_runtime; /* wraps Chorus::ChorusRuntime  */
typedef struct chorus_options chorus_options; /* wraps Chorus::OptionMap      */
typedef struct chorus_request chorus_request; /* wraps Chorus::GenerationRequest */

typedef int64_t chorus_request_id; /* -1 = invalid / none */

/* ========================================================================
 * Enums -- mirror include/chorus/core exactly; keep in sync by hand until
 * generated. Explicit values: these are ABI.
 * ======================================================================== */

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

typedef enum chorus_backend {
    CHORUS_BACKEND_LLAMA = 0,
    CHORUS_BACKEND_ECHO = 1,
} chorus_backend;

typedef enum chorus_event_kind {
    CHORUS_EVENT_TOKEN = 0,
    CHORUS_EVENT_REASONING_TOKEN = 1,
    CHORUS_EVENT_COMPLETE = 2,
    CHORUS_EVENT_ERROR = 3,
    CHORUS_EVENT_HISTORY_TRUNCATED = 4,
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
} chorus_log_level;

/* ========================================================================
 * Plain structs -- POD only, pointers borrowed unless stated.
 * ======================================================================== */

/* Mirrors Chorus::ChatMessage. */
typedef struct chorus_chat_message {
    const char* role;
    const char* content;
} chorus_chat_message;

/* Mirrors Chorus::RuntimeEvent. Produced by chorus_poll; all pointers are
 * owned by the runtime and valid until the next chorus_poll or
 * chorus_runtime_free on the same handle. session is NULL for stateless
 * requests. text: token chunk / full text on COMPLETE / message on ERROR.
 * reasoning: full accumulated reasoning on COMPLETE, else NULL. dropped:
 * history messages dropped, meaningful on HISTORY_TRUNCATED. */
typedef struct chorus_event {
    chorus_event_kind kind;
    chorus_request_id request_id;
    const char* session;
    const char* text;
    chorus_error error;
    const char* reasoning;
    int32_t dropped;
} chorus_event;

/* Log sink, installed at runtime creation. May be called during any chorus_*
 * call on the runtime's thread; must not call back into the runtime. */
typedef void (*chorus_log_callback)(chorus_log_level level, const char* message, void* user_data);

/* ========================================================================
 * Library
 * ======================================================================== */

/* ABI version of this header; bump on any breaking change. */
CHORUS_API uint32_t chorus_abi_version(void);

/* Static name for an error code, e.g. "SessionBusy". Never NULL. */
CHORUS_API const char* chorus_error_name(chorus_error error);

/* Frees a char* returned by any chorus_* function documented as caller-owned.
 * NULL is a no-op. */
CHORUS_API void chorus_string_free(char* str);

/* ========================================================================
 * Runtime lifecycle
 * ======================================================================== */

/* log_callback may be NULL (stderr fallback). Returns NULL on allocation
 * failure only. */
CHORUS_API chorus_runtime* chorus_runtime_new(chorus_log_callback log_callback, void* log_user_data);

/* Stops the engine (as chorus_stop_all) and releases everything, including
 * any event array from the last poll. NULL is a no-op. */
CHORUS_API void chorus_runtime_free(chorus_runtime* rt);

/* Detail for the most recent failed call on this runtime. Owned by the
 * runtime, valid until the next fallible call. Empty string, never NULL. */
CHORUS_API const char* chorus_last_error_message(const chorus_runtime* rt);

/* ========================================================================
 * Engine loading -- the composition root. The shim selects the backend via
 * the factory and injects it; consumers never see an engine.
 * ======================================================================== */

/* Load-time backend options (context_size, gpu_layers, ...). Key vocabulary
 * is the backend's; unknown keys are rejected at load, never dropped.
 * TODO(deviations #1): enumerate keys from provider self-description once
 * the contract exposes option metadata, instead of documenting them here. */
CHORUS_API chorus_options* chorus_options_new(void);
CHORUS_API void chorus_options_free(chorus_options* opts);
CHORUS_API void chorus_options_set_int(chorus_options* opts, const char* key, int64_t value);
CHORUS_API void chorus_options_set_float(chorus_options* opts, const char* key, double value);
CHORUS_API void chorus_options_set_bool(chorus_options* opts, const char* key, bool value);
CHORUS_API void chorus_options_set_string(chorus_options* opts, const char* key, const char* value);

/* Builds the engine and loads the model. Replaces any loaded engine (live
 * requests get Cancelled terminals; old engine torn down before the new one
 * initializes). model_path: .gguf path; ignored for ECHO (pass NULL).
 * options: borrowed, may be NULL for backend defaults. */
CHORUS_API chorus_error
chorus_load(chorus_runtime* rt, chorus_backend backend, const char* model_path, const chorus_options* options);

CHORUS_API bool chorus_is_loaded(const chorus_runtime* rt);

/* Stops and destroys the engine. Live requests receive exactly one Cancelled
 * terminal on a later chorus_poll. */
CHORUS_API void chorus_stop_all(chorus_runtime* rt);

/* ========================================================================
 * Requests -- flat builder over GenerationRequest + GenerationConfig,
 * matching the product's request vocabulary. A builder is reusable across
 * submits and is never consumed by chorus_generate.
 * ======================================================================== */

CHORUS_API chorus_request* chorus_request_new(void);
CHORUS_API void chorus_request_free(chorus_request* req);

/* Identity and delivery. session NULL or unset = stateless; empty string is
 * InvalidRequest at submit. stream=false suppresses TOKEN events. */
CHORUS_API void chorus_request_set_prompt(chorus_request* req, const char* prompt);
CHORUS_API void chorus_request_set_session(chorus_request* req, const char* session);
CHORUS_API void chorus_request_set_priority(chorus_request* req, int32_t priority);
CHORUS_API void chorus_request_set_stream(chorus_request* req, bool stream);

/* Portable generation config; unset = backend default. */
CHORUS_API void chorus_request_set_max_tokens(chorus_request* req, int32_t max_tokens);
CHORUS_API void chorus_request_set_temperature(chorus_request* req, float temperature);
CHORUS_API void chorus_request_set_top_k(chorus_request* req, int32_t top_k);
CHORUS_API void chorus_request_set_top_p(chorus_request* req, float top_p);
CHORUS_API void chorus_request_set_seed(chorus_request* req, uint64_t seed);
CHORUS_API void chorus_request_set_frequency_penalty(chorus_request* req, float penalty);
CHORUS_API void chorus_request_set_presence_penalty(chorus_request* req, float penalty);
CHORUS_API void chorus_request_add_stop(chorus_request* req, const char* sequence);
CHORUS_API void chorus_request_set_constraint(chorus_request* req, chorus_constraint_format format, const char* source);
CHORUS_API void chorus_request_set_thinking(chorus_request* req, bool thinking);

/* Backend-specific generation options, e.g. ("llama", "repeat_penalty"). */
CHORUS_API void
chorus_request_set_backend_option_float(chorus_request* req, const char* backend, const char* key, double value);
CHORUS_API void
chorus_request_set_backend_option_int(chorus_request* req, const char* backend, const char* key, int64_t value);

/* Chat controls; sessioned requests only, rejected on stateless ones.
 * inject: ephemeral message spliced into the fitted copy, depth counted from
 * the end (0 = just before the assistant prefix). chat_template: override;
 * unset = model's embedded template. */
CHORUS_API void chorus_request_add_inject(chorus_request* req, chorus_chat_message message, int32_t depth);
CHORUS_API void chorus_request_set_chat_template(chorus_request* req, const char* chat_template);

/* ========================================================================
 * Generation
 * ======================================================================== */

/* Submit. On CHORUS_OK, *out_request_id is the live request's id; otherwise
 * no request was created, *out_request_id is -1, and
 * chorus_last_error_message has the rejection detail. */
CHORUS_API chorus_error
chorus_generate(chorus_runtime* rt, const chorus_request* req, chorus_request_id* out_request_id);

/* Reroll the session's last assistant reply. req must have an empty prompt
 * and a session; other fields apply to the rerolled turn. */
CHORUS_API chorus_error
chorus_regenerate(chorus_runtime* rt, const chorus_request* req, chorus_request_id* out_request_id);

/* Requests stay active until chorus_poll drains their terminal event. */
CHORUS_API bool chorus_cancel(chorus_runtime* rt, chorus_request_id request_id);
CHORUS_API bool chorus_is_request_active(const chorus_runtime* rt, chorus_request_id request_id);
CHORUS_API chorus_request_id chorus_active_request_for_session(const chorus_runtime* rt, const char* session);

/* Drains pending events. Returns a runtime-owned array of *out_count events,
 * valid until the next chorus_poll or chorus_runtime_free; never NULL
 * (*out_count = 0 when idle). Call once per tick: pending events are retained
 * without bound until drained. */
CHORUS_API const chorus_event* chorus_poll(chorus_runtime* rt, size_t* out_count);

/* ========================================================================
 * Conversation history
 * ======================================================================== */

/* history: borrowed array of count messages, copied. Fails with SessionBusy
 * mid-turn. */
CHORUS_API chorus_error
chorus_history_import(chorus_runtime* rt, const char* session, const chorus_chat_message* history, size_t count);

/* Caller-owned snapshot: free with chorus_chat_messages_free. *out_count = 0
 * and NULL for an unknown session. */
CHORUS_API chorus_error chorus_history_export(
    const chorus_runtime* rt, const char* session, chorus_chat_message** out_messages, size_t* out_count
);
CHORUS_API void chorus_chat_messages_free(chorus_chat_message* messages, size_t count);

CHORUS_API chorus_error chorus_history_clear(chorus_runtime* rt, const char* session);

/* Rewrite one message's content in place. index negative = from the end
 * (-1 = last). TODO(deviations #3): forwards to the runtime's edit verb once
 * it exists; until then the shim composes export/mutate/import like the
 * Godot adapter does. */
CHORUS_API chorus_error
chorus_history_edit_message(chorus_runtime* rt, const char* session, int64_t index, const char* content);

/* Caller-owned array of session-id strings; free with chorus_string_list_free. */
CHORUS_API chorus_error chorus_list_conversations(const chorus_runtime* rt, char*** out_sessions, size_t* out_count);
CHORUS_API void chorus_string_list_free(char** strings, size_t count);

/* Drops every session lane. Fails with SessionBusy while any request lives. */
CHORUS_API chorus_error chorus_reset_context(chorus_runtime* rt);

CHORUS_API chorus_turn_outcome chorus_last_turn_outcome(const chorus_runtime* rt, const char* session);

/* The exact fitted prompt generation would consume for this session right
 * now, without generating. req carries the overrides (chat_template, inject,
 * config) and must match what you would generate with; may be NULL for
 * defaults. Returns a caller-owned string (chorus_string_free), or NULL for
 * unknown session / no engine / no backend rendering. */
CHORUS_API char* chorus_render_prompt(const chorus_runtime* rt, const char* session, const chorus_request* req);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHORUS_C_H */
