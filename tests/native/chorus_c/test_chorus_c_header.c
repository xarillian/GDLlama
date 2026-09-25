#include "chorus_c/chorus_c.h"

#include <string.h>

int chorus_c_header_smoke(void) {
    chorus_error (*generate)(chorus_runtime*, const chorus_request*, chorus_submit_result*) = chorus_generate;
    chorus_error (*embed)(chorus_runtime*, const chorus_embedding_request*, chorus_submit_result*) = chorus_embed;
    chorus_error (*load)(chorus_runtime*, chorus_provider, const char*, const chorus_options*, chorus_log_level, chorus_load_result*) = chorus_load;
    bool (*cancel_load)(chorus_runtime*, chorus_load_id) = chorus_cancel_load;
    chorus_load_id (*active_load)(const chorus_runtime*) = chorus_active_load_id;
    chorus_error (*render)(chorus_runtime*, const chorus_request*, chorus_submit_result*) = chorus_render_prompt;

    if (chorus_abi_version() != 6 || strcmp(chorus_error_name(CHORUS_ERR_SESSION_BUSY), "SessionBusy") != 0)
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

    chorus_runtime_free(NULL);
    chorus_options_free(NULL);
    chorus_request_free(NULL);
    chorus_conversation_messages_free(NULL, 0);
    chorus_string_list_free(NULL, 0);
    return 0;
}
