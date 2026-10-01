#pragma once

#include "chorus/core/log.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Keeps every record a Logger produces, for tests that assert on level,
// message, and fields rather than on presentation.
class CollectingLog {
  public:
    CollectingLog() : _channel(std::make_shared<Chorus::LogChannel>(4096)) {}

    // A logger writing here, hearing everything unless told otherwise.
    Chorus::Logger logger(Chorus::LogLevel minimum = Chorus::LogLevel::Debug) {
        return Chorus::Logger(_channel, minimum);
    }

    std::vector<Chorus::LogRecord> records() const {
        collect();
        std::lock_guard<std::mutex> lock(_mutex);
        return _records;
    }

    size_t size() const { return records().size(); }

    // How many records carry `message` verbatim. Verbatim on purpose: a
    // message with an interpolated value in it is the defect this asserts away.
    size_t count(const std::string& message) const {
        const auto snapshot = records();
        size_t seen = 0;
        for (const auto& record : snapshot)
            seen += record.message == message ? 1 : 0;
        return seen;
    }

    void clear() {
        _channel->drain();
        std::lock_guard<std::mutex> lock(_mutex);
        _records.clear();
    }

  private:
    void collect() const {
        auto fresh = _channel->drain();
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto& record : fresh)
            _records.push_back(std::move(record));
    }

    std::shared_ptr<Chorus::LogChannel> _channel;
    mutable std::mutex _mutex;
    mutable std::vector<Chorus::LogRecord> _records;
};

/// The value of `key` on `record`, or nullptr when it carries no such field.
inline const Chorus::LogValue* find_log_field(const Chorus::LogRecord& record, const std::string& key) {
    for (const auto& field : record.fields) {
        if (field.first == key)
            return &field.second;
    }
    return nullptr;
}
