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

/*
 * Adapts the host-neutral Chorus runtime to a Godot node.
 *
 * `GodotChorus` translates Godot values into the typed runtime API, constructs
 * the selected provider through the factory, and delivers asynchronous runtime
 * events as signals on the Godot thread. Provider-declared capabilities drive
 * the Inspector surface, so this adapter does not duplicate provider schemas.
 */
class GodotChorus : public godot::Node {
    GDCLASS(GodotChorus, godot::Node);

  protected:
    static void _bind_methods();
    void _notification(int p_what);

    // The selected provider owns its option names, types, defaults, and bounds.
    // These hooks expose that declaration as Inspector properties without
    // hard-coding a provider's configuration surface in `GodotChorus`.
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
        PROVIDER_LLAMA,
        PROVIDER_ECHO,
    };

    enum TurnOutcomeCode {
        TURN_NONE,
        TURN_COMPLETED,
        TURN_CANCELLED,
        TURN_ERRORED,
    };

    // `GodotChorus::LOG_OFF` is a threshold only and is never emitted.
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

    /*
     * Declares `chorus/logging/min_level` in Godot Project Settings.
     *
     * Module initialization calls this once. Editor runs default to
     * `GodotChorus::LOG_INFO`; exported games use `Chorus::log_level_default`
     * because a player's console is not a debug channel.
     */
    static void register_project_settings();

    /*
     * Constructs the selected provider and loads it into the runtime.
     *
     * Every call replaces the current engine so the complete node
     * configuration takes effect. Replacement terminates in-flight requests
     * with `Chorus::ChorusError::Cancelled`.
     */
    bool load_model();
    void stop_all();
    bool is_loaded() const;
    bool supports_embeddings() const;

    /*
     * Submits one stateless generation or sessioned chat turn.
     *
     * `request` accepts these request fields:
     *  - `prompt`: required `godot::String`.
     *  - `stream`: emits `GodotChorus::token_generated` for each token when true.
     *  - `priority`: higher values enter the runtime queue first.
     *  - `session`: stable continuity lane; empty or absent means stateless.
     *
     * A session permits one live request. After `GodotChorus::stop_all` or
     * `GodotChorus::load_model`, resubmission remains busy until
     * `GodotChorus::_process` drains the cancelled terminal event, preserving
     * per-session event order.
     *
     * Generation fields overlay `GodotChorus::generation_defaults`, which
     * overlays the engine defaults. An absent field inherits, `null` clears an
     * inherited value, and any other value replaces it:
     *  - `max_tokens`, `top_k`: integer values.
     *  - `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`: floating-point values.
     *  - `seed`: non-negative integer.
     *  - `stop`: array of strings; an empty array disables inherited stop sequences.
     *  - `provider_options`: recursively merged namespaces; a `null` leaf erases its key.
     *  - `repeat_penalty`: shorthand for `provider_options["llama"]["repeat_penalty"]`.
     *  - `constraint`: dictionary containing `format` and `source`.
     *  - `grammar`, `json_schema`, `json`: mutually exclusive constraint shorthands.
     *  - `show_thinking`: reasoning-model toggle.
     *
     * `repeat_penalty` is applied after `provider_options`, so the shorthand
     * wins when both specify the same option. Unknown provider options and
     * multiple constraint spellings reject the request.
     *
     * Sessioned requests also accept:
     *  - `inject`: ephemeral `{role, content, depth?}` messages excluded from durable history.
     *  - `chat_template`: per-request template overriding `GodotChorus::chat_template`.
     *
     * Returns:
     *  - `int64_t`: the non-negative accepted request ID.
     *  - `-1`: normalization or submission failed.
     */
    int64_t generate(const godot::Dictionary& request);
    int64_t embed(const godot::String& prompt, int64_t priority = 0);

    /*
     * Rerolls the last assistant message in a session.
     *
     * `overrides` accepts the generation fields from `GodotChorus::generate`
     * except `prompt`; history supplies the turn text. The positional
     * `session` argument selects the lane, so a dictionary `session` field is
     * ignored. Completion replaces the old reply, while cancellation or error
     * restores it.
     *
     * Returns:
     *  - `int64_t`: the non-negative accepted request ID.
     *  - `-1`: normalization or submission failed.
     */
    int64_t regenerate(const godot::String& session, const godot::Dictionary& overrides);

    /// Requests remain active until `GodotChorus::_process` drains their terminal event.
    bool cancel_request(int64_t request_id);
    /// Returns true until `GodotChorus::_process` drains the terminal event.
    bool is_request_active(int64_t request_id) const;
    /*
     * Returns:
     *  - `int64_t`: the non-negative active request ID.
     *  - `-1`: the session is unknown or has no active request.
     */
    int64_t active_request_for_session(const godot::String& session) const;

    /*
     * Replaces one session's durable history.
     *
     * Validation and runtime failures return false and are reported through
     * Godot's error log.
     *
     * Errors:
     *  - `Chorus::ChorusError::SessionBusy`: the session has active work.
     */
    bool import_conversation_history(const godot::String& session, const godot::Array& history);
    /*
     * Returns:
     *  - `godot::Array`: `{role, content}` dictionaries for the session.
     *  - Empty `godot::Array`: the session is unknown.
     */
    godot::Array export_conversation_history(const godot::String& session) const;
    /*
     * Clears one session's durable history.
     *
     * Runtime failures return false and are reported through Godot's error log.
     *
     * Errors:
     *  - `Chorus::ChorusError::SessionBusy`: the session has active work.
     */
    bool clear_conversation_history(const godot::String& session);
    /*
     * Rewrites one message without changing its role.
     *
     * Negative indexes count from the end, with `-1` naming the newest
     * message. Unknown sessions, out-of-range indexes, and busy sessions fail.
     * Role changes and message insertion or deletion use an export, mutate,
     * import round trip.
     */
    bool edit_message(const godot::String& session, int64_t index, const godot::String& content);
    godot::PackedStringArray list_conversations() const;
    /*
     * Clears every conversation.
     *
     * Runtime failures return false and are reported through Godot's error log.
     *
     * Errors:
     *  - `Chorus::ChorusError::SessionBusy`: at least one session has active work.
     */
    bool reset_context();
    TurnOutcomeCode last_turn_outcome(const godot::String& session) const;
    /*
     * Renders the prompt a default-configured turn would consume now.
     *
     * The node's effective generation defaults determine the fitting
     * reservation and `show_thinking` value. A later `GodotChorus::generate`
     * call can fit differently by overriding either value.
     *
     * Returns:
     *  - `godot::String`: the fitted prompt when rendering is available.
     *  - Empty `godot::String`: rendering is unavailable.
     */
    godot::String render_chat_prompt(
        const godot::String& session,
        const godot::String& template_override = godot::String(),
        const godot::Array& inject = godot::Array()
    );

    void _process(double delta) override;

    void set_model_path(const godot::String& path);
    godot::String get_model_path() const;
    void set_provider(ProviderChoice provider);
    ProviderChoice get_provider() const;
    void set_generation_defaults(const godot::Ref<ChorusGenerationDefaults>& defaults);
    godot::Ref<ChorusGenerationDefaults> get_generation_defaults() const;
    /*
     * Sets the node-level Jinja chat template for sessioned turns.
     *
     * An empty string selects the model's embedded template. Stateless
     * `GodotChorus::generate` calls ignore this setting. The Echo provider
     * accepts it as inert because content controls are vacuous on the test
     * double.
     */
    void set_chat_template(const godot::String& chat_template);
    godot::String get_chat_template() const;
    /*
     * Selects the least severe `Chorus::LogLevel` for the next engine.
     *
     * When the override is disabled, `chorus/logging/min_level` supplies the
     * value. The setting takes effect on the next `GodotChorus::load_model`
     * because it belongs to the engine configuration.
     */
    void set_override_log_level(bool enabled);
    bool get_override_log_level() const;
    void set_log_level(int64_t level);
    int64_t get_log_level() const;

    /*
     * Computes cosine similarity between two Godot arrays.
     *
     * Inputs must have equal, non-zero lengths. A zero-magnitude vector has no
     * usable direction and also produces zero.
     *
     * Returns:
     *  - `float`: the cosine similarity for valid vectors.
     *  - `0.0f`: the inputs are invalid or either vector has zero magnitude.
     */
    float similarity_cos(godot::PackedFloat32Array array1, godot::PackedFloat32Array array2) const;

  private:
    // Returns the assigned resource or lazily constructs an internal resource
    // with `max_tokens` set to 128, ensuring the host layer always contributes
    // generation defaults.
    godot::Ref<ChorusGenerationDefaults> effective_generation_defaults();

    // Sends the node's ambient settings to the runtime before every submission
    // and render operation, leaving the runtime to decide where each value
    // applies.
    bool push_host_defaults();

    void drain_logs();
    Chorus::LogLevel effective_log_level() const;

    // Caches the selected provider's self-description because Godot requests
    // the Inspector property list frequently.
    const Chorus::EngineCapabilities& provider_capabilities() const;
    const Chorus::ProviderOptionDescriptors& load_option_descriptors() const;
    const Chorus::ProviderOptionDescriptor* find_load_option(const godot::StringName& name) const;

    Chorus::ChorusRuntime _runtime;

    godot::String _model_path;

    ProviderChoice _provider = PROVIDER_LLAMA;

    // Contains only values explicitly set by the user. Other values resolve
    // from provider defaults at load time. Options for unselected providers
    // remain inert in memory, preserving them when switching providers during
    // a session. Only the selected provider's listed properties persist in a
    // saved scene.
    Chorus::ProviderOptionMap _load_options;
    mutable Chorus::EngineCapabilities _cached_capabilities;
    mutable std::optional<ProviderChoice> _cached_capabilities_provider;

    godot::String _chat_template;

    bool _override_log_level = false;
    int64_t _log_level = (int64_t)Chorus::log_level_default;

    // Remains null until the user assigns a resource, keeping scene
    // serialization clean. `GodotChorus::effective_generation_defaults`
    // supplies the internal fallback.
    godot::Ref<ChorusGenerationDefaults> _generation_defaults;
    godot::Ref<ChorusGenerationDefaults> _fallback_generation_defaults;
};

VARIANT_ENUM_CAST(GodotChorus::ErrorCode);
VARIANT_ENUM_CAST(GodotChorus::ProviderChoice);
VARIANT_ENUM_CAST(GodotChorus::TurnOutcomeCode);
VARIANT_ENUM_CAST(GodotChorus::LogLevelCode);
