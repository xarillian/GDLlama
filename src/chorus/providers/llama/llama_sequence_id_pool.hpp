#pragma once

#include <optional>
#include <set>
#include <stdexcept>

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

  private:
    std::set<int> _free;
    std::set<int> _allocated;
};

} // namespace Chorus
