#include "chorus_c/chorus_c.h"

#include <type_traits>

static_assert(std::is_same_v<decltype(chorus_log_record::produced_at), double>);

void compile_c_log_record_shape() {
    chorus_log_record record{};
    auto [level, message, fields, field_count, request_id, session, produced_at] = record;
    (void)level;
    (void)message;
    (void)fields;
    (void)field_count;
    (void)request_id;
    (void)session;
    (void)produced_at;
}
