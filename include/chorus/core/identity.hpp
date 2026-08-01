#pragma once

#include <cstdint>
#include <string>

namespace Chorus {

/*
 * The names the system uses to talk about work.
 *
 * They live alone, below every other core header, because both the request
 * vocabulary and the diagnostics channel name work and neither may include the
 * other.
 */

/// Runtime-assigned, unique for the life of one `ChorusRuntime`.
using RequestId = int64_t;

/// Caller-owned continuity lane (an NPC, a dialogue thread). Never empty when present.
using SessionId = std::string;

} // namespace Chorus
