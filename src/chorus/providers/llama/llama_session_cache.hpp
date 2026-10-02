#pragma once

#include "chorus/core/common.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace Chorus {

/*
 * Tracks conversations whose KV state stays resident in idle sequence slots between turns.
 *
 * Holds bookkeeping only; the caller owns the KV data and moves or clears it alongside
 * each change recorded here. Each session is parked in at most one slot.
 */
class LlamaSessionCache {
  public:
    struct Entry {
        SessionId session;
        std::vector<int32_t> tokens;
        uint64_t parked_at = 0;
    };

    /*
     * Records `tokens` as the KV contents of idle `slot` for `session`.
     *
     * Returns:
     *  - `int`: a different slot that held an older copy of `session`, whose KV the caller must clear.
     *  - `std::nullopt`: no older copy existed.
     */
    std::optional<int> park(int slot, SessionId session, std::vector<int32_t> tokens);

    std::optional<int> slot_for(const SessionId& session) const;
    bool holds(int slot) const;
    std::size_t token_count(int slot) const;

    /// Removes and returns the entry parked in `slot`, which must hold one.
    Entry take(int slot);

    /// Records that the KV parked in `from` now lives in the empty slot `to`.
    void move(int from, int to);

    /// Returns the slot whose session has been parked longest.
    std::optional<int> least_recent() const;

    void clear() { _entries.clear(); }

  private:
    std::map<int, Entry> _entries;
    uint64_t _clock = 0;
};

} // namespace Chorus
