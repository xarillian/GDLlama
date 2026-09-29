#pragma once

#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace Chorus {

class LlamaSequenceIdPool {
  public:
    explicit LlamaSequenceIdPool(int count = 0) {
        reset(count);
    }

    void reset(int count = 0) {
        if (count < 0)
            throw std::logic_error("sequence identifier count cannot be negative");
        _free.clear();
        _allocated.clear();
        for (int id = 0; id < count; ++id)
            _free.insert(id);
    }

    std::optional<int> acquire() {
        if (_free.empty())
            return std::nullopt;
        const int id = *_free.begin();
        _free.erase(_free.begin());
        _allocated.insert(id);
        return id;
    }

    void release(int id) {
        if (!_allocated.erase(id))
            throw std::logic_error("unknown or released sequence identifier");
        _free.insert(id);
    }

    bool empty() const { return _free.empty(); }

    /// Returns the highest allocated identifier and the lowest free one when the free one is lower.
    std::optional<std::pair<int, int>> compaction_move() const {
        if (_free.empty() || _allocated.empty() || *_free.begin() > *_allocated.rbegin())
            return std::nullopt;
        return std::pair{*_allocated.rbegin(), *_free.begin()};
    }

    void move(int from, int to) {
        if (!_allocated.contains(from) || !_free.contains(to))
            throw std::logic_error("sequence identifier move requires an allocated source and a free destination");
        _allocated.erase(from);
        _free.erase(to);
        _allocated.insert(to);
        _free.insert(from);
    }

  private:
    std::set<int> _free;
    std::set<int> _allocated;
};

} // namespace Chorus
