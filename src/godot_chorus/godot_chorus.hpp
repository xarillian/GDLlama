#pragma once

#include <optional>

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/templates/list.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include "chorus/core/capabilities.hpp"
#include "chorus/core/common.hpp"
#include "chorus/runtime/runtime.hpp"
#include "godot_chorus/chorus_generation_defaults.hpp"

class GodotChorus : public godot::Node {
    GDCLASS(GodotChorus, godot::Node);

  protected:
    static void _bind_methods();
    void _notification(int p_what);

    // --- Provider load options, rendered from the provider's declared schema ---
    // The selected provider owns its option names, types, defaults, and bounds;
    // these hooks restate that declaration as inspector properties so the node
    // never hard-codes a provider's configuration surface.
    bool _set(const godot::StringName& name, const godot::Variant& value);
    bool _get(const godot::StringName& name, godot::Variant& ret) const;
    void _get_property_list(godot::List<godot::PropertyInfo>* list) const;
    bool _property_can_revert(const godot::StringName& name) const;
    bool _property_get_revert(const godot::StringName& name, godot::Variant& ret) const;

  public:
    enum ErrorCode {
        ERR_NONE,
        ERR_MODEL_LOAD,
        ERR_CONTEXT_INIT,
        ERR_DECODE,
        ERR_TOKENIZE,
        ERR_INVALID_REQUEST,
        ERR_ENGINE_NOT_READY,
        ERR_CANCELLED,
        ERR_UNSUPPORTED_MODEL_FORMAT,
        ERR_UNSUPPORTED_FEATURE,
        ERR_UNSUPPORTED_OPTION,
        ERR_SESSION_BUSY,
        ERR_UNKNOWN,
    };

    enum ProviderChoice {
        PROVIDER_LLAMA, // mirrors Chorus::Provider::Llama
        PROVIDER_ECHO,  // mirrors Chorus::Provider::Echo
    };

    // Mirrors Chorus::TurnOutcome: the terminal state of a session's most
    // recent chat turn.
    enum TurnOutcomeCode {
        TURN_NONE,
        TURN_COMPLETED,
        TURN_CANCELLED,
        TURN_ERRORED,
    };

    // Mirrors Chorus::LogLevel: the severity a log_message signal carries, and
    // the verbosity a node asks for. LOG_OFF is a threshold only; no record
    // arrives carrying it.
    enum LogLevelCode {
        LOG_DEBUG,
        LOG_INFO,
        LOG_WARN,
        LOG_ERROR,
        LOG_FATAL,
        LOG_OFF,
    };

    static int to_godot(Chorus::ChorusError e);
    static int to_godot(Chorus::TurnOutcome outcome);
    static int to_godot(Chorus::LogLevel level);

    // Declares chorus/logging/min_level so it appears in Project Settings, which
    // is where a Godot developer looks for verbosity. Called once at module
    // initialization; the editor gets Info and above, an export template Warn
    // and above, because a player's console is not a debug channel.
    static void register_project_settings();

    // --- Core API ---

    // Constructs the selected provider via the factory and hands it to the
    // runtime. Always (re)loads: a second call replaces the engine (in-flight
    // requests get ERR_CANCELLED), and the current config fully applies.
    bool load_model();
    void stop_all();
    bool is_loaded() const;

    // Request dict keys:
    //   prompt: String       (required, must not be null)
    //   stream: bool         (If true, token_generated fires per token. Default: false)
    //   priority: int        (Higher = earlier in queue. Default: 0)
    //   session: String      (stable continuity lane, e.g. "npc_42/dialogue"; one live request per
    //                          session; empty/omitted = stateless. A same-frame resubmit after
    //                          stop_all()/load_model() is SessionBusy until poll() drains the Cancelled
    //                          terminal; deliberate, to preserve per-session event ordering.)
    //
    // Generation overlay keys layer onto generation_defaults (the assigned resource, else an
    // internal default carrying max_tokens = 128), which itself layers onto the engine's
    // defaults. Absent = inherit the layer below; an explicit null clears an inherited value;
    // a present non-null value replaces it:
    //   max_tokens, temperature, top_k, top_p, seed, frequency_penalty, presence_penalty
    //   stop: Array[String]  (null clears to the provider default; [] explicitly disables any
    //                          inherited stop sequences; a non-empty array replaces them)
    //   provider_options: Dictionary
    //                        (namespaced, e.g. {"llama": {"repeat_penalty": 1.1}}; deep-merges
    //                          onto the inherited provider options; a null leaf erases the
    //                          corresponding inherited key; unknown options are rejected)
    //   repeat_penalty: float (convenience for provider_options["llama"]["repeat_penalty"]; applied
    //                          after provider_options, so it wins if both are given; null erases it)
    //   constraint: Dictionary {"format": "gbnf"|"json_schema", "source": String}, or one of the
    //                          convenience spellings grammar: String (GBNF text) / json_schema:
    //                          String (schema text) / json: String (same as json_schema); at most
    //                          one spelling may be present; null clears an inherited constraint.
    //   thinking: bool       (reasoning-model toggle; null clears an inherited value back to the
    //                          template/provider default)
    // Chat keys (meaningful on sessioned requests):
    //   inject: Array        (of {role: String, content: String, depth?: int} Dictionaries;
    //                          ephemeral messages placed into this turn's prompt only, never
    //                          durable history; depth counts from the end, 0 = last)
    //   chat_template: String (per-request jinja override; wins over the chat_template node
    //                          property; null reads as absent)
    // Returns the request ID (>= 0) on success, or -1 on failure.
    int64_t generate(const godot::Dictionary& request);

    // Reroll the session's last assistant line. `overrides` takes the same
    // keys as generate() minus 'prompt' (the turn text comes from history);
    // a 'session' key in it is ignored -- the positional argument names the
    // lane. On completion the new reply replaces the old; on cancel/error the
    // old reply is restored. Returns the request ID (>= 0), or -1 on failure.
    int64_t regenerate(const godot::String& session, const godot::Dictionary& overrides);

    // --- Runtime controls ---

    // Requests stay active until _process() drains their terminal event.
    bool cancel_request(int64_t request_id);
    bool is_request_active(int64_t request_id) const;
    // -1 if the session has no active request (including an unknown session).
    int64_t active_request_for_session(const godot::String& session) const;

    // --- Conversation history ---
    // import/clear on a busy session (and reset_context with ANY busy session)
    // fail with SessionBusy: cancel or stop_all + poll first.

    bool import_conversation_history(const godot::String& session, const godot::Array& history);
    // Array of {role, content} Dictionaries; empty for an unknown session.
    godot::Array export_conversation_history(const godot::String& session) const;
    bool clear_conversation_history(const godot::String& session);
    // Rewrites one message's content in place (any role). A negative index
    // counts from the end (-1 = newest); an unknown session or out-of-range
    // index fails, as does a busy session. Role edits and insert/delete stay
    // on the export -> mutate -> import roundtrip.
    bool edit_message(const godot::String& session, int64_t index, const godot::String& content);
    godot::PackedStringArray list_conversations() const;
    bool reset_context();
    TurnOutcomeCode last_turn_outcome(const godot::String& session) const;
    // The exact fitted prompt generation would consume for this session right
    // now ("" when unavailable: unknown session, no engine, or no provider
    // rendering). Uses this node's effective generation defaults for the
    // fitting reservation and thinking flag, so inspection matches a
    // default-configured turn; a generate() call overriding max_tokens or
    // thinking per-request can still fit differently.
    godot::String render_chat_prompt(
        const godot::String& session,
        const godot::String& template_override = godot::String(),
        const godot::Array& inject = godot::Array()
    );

    void _process(double delta) override;

    // --- Properties ---

    void set_model_path(const godot::String& path);
    godot::String get_model_path() const;
    void set_provider(ProviderChoice provider);
    ProviderChoice get_provider() const;
    void set_generation_defaults(const godot::Ref<ChorusGenerationDefaults>& defaults);
    godot::Ref<ChorusGenerationDefaults> get_generation_defaults() const;
    // Node-default jinja chat template for chat (sessioned) turns; "" = the
    // model's embedded template. Stateless generate() calls ignore it. Echo
    // accepts it as inert (content controls are vacuous on the test double).
    void set_chat_template(const godot::String& chat_template);
    godot::String get_chat_template() const;
    // Verbosity for the engine this node loads: the least severe LogLevelCode
    // worth reporting. The pair is the usual override idiom, with the
    // chorus/logging/min_level project setting deciding when the override is
    // off. Takes effect at the next load_model(), since the level travels with
    // the engine's config.
    void set_override_log_level(bool enabled);
    bool get_override_log_level() const;
    void set_log_level(int64_t level);
    int64_t get_log_level() const;

    // --- Utility ---
    float similarity_cos(godot::PackedFloat32Array array1, godot::PackedFloat32Array array2) const;

  private:
    // The assigned resource when set, otherwise a lazily constructed internal
    // default instance (max_tokens = 128), so the middle layer always
    // contributes a generation default even with the property unset.
    godot::Ref<ChorusGenerationDefaults> effective_generation_defaults();

    // Hands this node's ambient settings to the runtime, which resolves them
    // against each request. Called from every entry point that submits or
    // renders, so the node never has to decide where an ambient value applies.
    void push_host_defaults();

    // Drains the runtime's log channel and presents it: console routing that
    // respects what each Godot channel means, plus the log_message signal for
    // projects that want their own presentation.
    void drain_logs();
    // The node's override when it is on, else the project setting.
    Chorus::LogLevel effective_log_level() const;

    // The selected provider's self-description, fetched from the factory and
    // cached because the inspector asks for the property list constantly.
    const Chorus::EngineCapabilities& provider_capabilities() const;
    const Chorus::ProviderOptionDescriptors& load_option_descriptors() const;
    const Chorus::ProviderOptionDescriptor* find_load_option(const godot::StringName& name) const;

    Chorus::ChorusRuntime _runtime;

    godot::String _model_path;

    ProviderChoice _provider = PROVIDER_LLAMA;

    // Only the options the user actually set; everything else resolves from
    // the provider's declared defaults at load time. Keys belonging to a
    // provider that is not currently selected are inert, so switching back and
    // forth in a session keeps a configuration. A scene save does not: only
    // the selected provider's options are listed as properties, so only they
    // persist.
    Chorus::ProviderOptionMap _load_options;
    mutable Chorus::EngineCapabilities _cached_capabilities;
    mutable std::optional<ProviderChoice> _cached_capabilities_provider;

    // Node-default jinja chat template; "" = the model's embedded template.
    godot::String _chat_template;

    bool _override_log_level = false;
    int64_t _log_level = (int64_t)Chorus::log_level_default;

    // The bound property: null when the user has not assigned a resource, so
    // scene serialization stays clean. effective_generation_defaults() supplies
    // the internal fallback when unset.
    godot::Ref<ChorusGenerationDefaults> _generation_defaults;
    godot::Ref<ChorusGenerationDefaults> _fallback_generation_defaults;
};

VARIANT_ENUM_CAST(GodotChorus::ErrorCode);
VARIANT_ENUM_CAST(GodotChorus::ProviderChoice);
VARIANT_ENUM_CAST(GodotChorus::TurnOutcomeCode);
VARIANT_ENUM_CAST(GodotChorus::LogLevelCode);
