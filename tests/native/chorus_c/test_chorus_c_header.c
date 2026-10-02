#include "chorus_c/chorus_c.h"

#include <string.h>

int chorus_c_header_smoke(void) {
    chorus_error (*generate)(chorus_runtime*, const chorus_request*, chorus_submit_result*) = chorus_generate;
    chorus_error (*embed)(chorus_runtime*, const chorus_embedding_request*, chorus_submit_result*) = chorus_embed;
    chorus_error (*load)(chorus_runtime*, chorus_provider, const char*, const chorus_options*, chorus_log_level, chorus_load_result*) = chorus_load;
    bool (*cancel_load)(chorus_runtime*, chorus_load_id) = chorus_cancel_load;
    chorus_load_id (*active_load)(const chorus_runtime*) = chorus_active_load_id;
    chorus_error (*render)(chorus_runtime*, const chorus_request*, chorus_submit_result*) = chorus_render_prompt;

    if (chorus_abi_version() != 9 || strcmp(chorus_error_name(CHORUS_ERR_SESSION_BUSY), "SessionBusy") != 0)
        return 1;
    if (chorus_request_set_prompt(NULL, "prompt") != CHORUS_ERR_INVALID_REQUEST)
        return 2;
    if (chorus_request_add_inject(NULL, CHORUS_ROLE_USER, "prompt", 0) != CHORUS_ERR_INVALID_REQUEST)
        return 3;
    chorus_load_result result = {0};
    if (load(NULL, CHORUS_PROVIDER_ECHO, NULL, NULL, CHORUS_LOG_OFF, &result) != CHORUS_ERR_INVALID_REQUEST || result.load_id != -1)
        return 8;
    if (cancel_load(NULL, 0) || active_load(NULL) != -1)
        return 9;
    if (generate(NULL, NULL, NULL) != CHORUS_ERR_INVALID_REQUEST)
        return 4;
    if (embed(NULL, NULL, NULL) != CHORUS_ERR_INVALID_REQUEST)
        return 5;
    if (render(NULL, NULL, NULL) != CHORUS_ERR_INVALID_REQUEST)
        return 6;

    chorus_error (*count)(chorus_runtime*, const char*, chorus_submit_result*) = chorus_count_message_tokens;
    if (count(NULL, NULL, NULL) != CHORUS_ERR_INVALID_REQUEST)
        return 7;

    if (chorus_request_clear_max_tokens(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_temperature(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_top_k(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_top_p(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_seed(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_frequency_penalty(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_presence_penalty(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_stop(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_set_empty_stop(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_constraint(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_set_unconstrained(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_show_thinking(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_chat_template(NULL) != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_provider_option(NULL, "echo", "key") != CHORUS_ERR_INVALID_REQUEST ||
        chorus_request_clear_provider_options(NULL) != CHORUS_ERR_INVALID_REQUEST)
        return 10;

    chorus_error (*load_defaults)(chorus_runtime*, const char*) = chorus_generation_defaults_load_file;
    chorus_error (*apply_defaults)(chorus_runtime*, const char*, size_t) = chorus_generation_defaults_apply_json;
    chorus_error (*export_defaults)(const chorus_runtime*, char**) = chorus_generation_defaults_export_json;
    chorus_error (*save_defaults)(chorus_runtime*, const char*) = chorus_generation_defaults_save_file;
    char* json = (char*)1;
    if (load_defaults(NULL, "path") != CHORUS_ERR_INVALID_REQUEST ||
        apply_defaults(NULL, "{}", 2) != CHORUS_ERR_INVALID_REQUEST ||
        export_defaults(NULL, &json) != CHORUS_ERR_INVALID_REQUEST || json != NULL ||
        save_defaults(NULL, "path") != CHORUS_ERR_INVALID_REQUEST)
        return 11;
    bool (*wait)(chorus_runtime*, uint64_t) = chorus_wait;
    if (wait(NULL, 0))
        return 12;
    chorus_runtime_free(NULL);
    chorus_options_free(NULL);
    chorus_request_free(NULL);
    chorus_conversation_messages_free(NULL, 0);
    chorus_string_list_free(NULL, 0);
    return 0;
}
