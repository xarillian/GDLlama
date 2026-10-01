#include "chorus/providers/llama/llama_session_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Chorus {

std::optional<int> LlamaSessionCache::park(int slot, SessionId session, std::vector<int32_t> tokens) {
    std::optional<int> stale = slot_for(session);
    if (stale == slot)
        stale.reset();
    if (stale)
        _entries.erase(*stale);
    _entries.insert_or_assign(slot, Entry{std::move(session), std::move(tokens), ++_clock});
    return stale;
}

std::optional<int> LlamaSessionCache::slot_for(const SessionId& session) const {
    const auto found =
        std::ranges::find_if(_entries, [&](const auto& entry) { return entry.second.session == session; });
    return found == _entries.end() ? std::nullopt : std::optional<int>{found->first};
}

bool LlamaSessionCache::holds(int slot) const {
    return _entries.contains(slot);
}

std::size_t LlamaSessionCache::token_count(int slot) const {
    const auto found = _entries.find(slot);
    return found == _entries.end() ? 0 : found->second.tokens.size();
}

LlamaSessionCache::Entry LlamaSessionCache::take(int slot) {
    auto node = _entries.extract(slot);
    if (node.empty())
        throw std::logic_error("no session is parked in this sequence slot");
    return std::move(node.mapped());
}

void LlamaSessionCache::move(int from, int to) {
    if (_entries.contains(to))
        throw std::logic_error("a parked session cannot move into an occupied sequence slot");
    auto node = _entries.extract(from);
    if (node.empty())
        throw std::logic_error("no session is parked in this sequence slot");
    node.key() = to;
    _entries.insert(std::move(node));
}

std::optional<int> LlamaSessionCache::least_recent() const {
    const auto found = std::ranges::min_element(_entries, {}, [](const auto& entry) { return entry.second.parked_at; });
    return found == _entries.end() ? std::nullopt : std::optional<int>{found->first};
}

} // namespace Chorus
