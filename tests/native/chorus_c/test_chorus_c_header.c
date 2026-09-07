#include "chorus_c/chorus_c.h"

#include <string.h>

int chorus_c_header_smoke(void) {
    chorus_error (*set_bool)(chorus_request*, const char*, const char*, bool) =
        chorus_request_set_provider_option_bool;
    chorus_error (*set_string)(chorus_request*, const char*, const char*, const char*) =
        chorus_request_set_provider_option_string;
    chorus_error (*embed)(chorus_runtime*, const char*, int32_t, chorus_execution_mode, chorus_request_id*) = chorus_embed;

    if (chorus_abi_version() != 3 || strcmp(chorus_error_name(CHORUS_ERR_SESSION_BUSY), "SessionBusy") != 0)
        return 1;
    if (set_bool(NULL, "echo", "flag", true) != CHORUS_ERR_INVALID_REQUEST)
        return 2;
    if (set_string(NULL, "echo", "mode", "strict") != CHORUS_ERR_INVALID_REQUEST)
        return 3;
    if (chorus_request_set_prompt(NULL, "prompt") != CHORUS_ERR_INVALID_REQUEST)
        return 4;
    if (chorus_options_set_string(NULL, "key", "value") != CHORUS_ERR_INVALID_REQUEST)
        return 5;
    if (embed(NULL, "prompt", 0, CHORUS_EXECUTION_SHARED, NULL) != CHORUS_ERR_INVALID_REQUEST)
        return 6;

    chorus_runtime_free(NULL);
    chorus_options_free(NULL);
    chorus_request_free(NULL);
    chorus_string_free(NULL);
    chorus_chat_messages_free(NULL, 0);
    chorus_string_list_free(NULL, 0);
    return 0;
}
