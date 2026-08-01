#pragma once

#include "chorus/core/log.hpp"

#include <mutex>
#include <string>
#include <vector>

// Keeps every record a Logger produces, for tests that assert on level,
// message, and fields rather than on formatted text.
//
// Locked because providers log from their own threads, which is the case worth
// exercising: a test that only ever logs inline proves nothing about the seam.
class CollectingSink {
  public:
    Chorus::LogSink sink() {
        return [this](Chorus::LogRecord record) {
            std::lock_guard<std::mutex> lock(_mutex);
            _records.push_back(std::move(record));
        };
    }

    // A logger writing here, hearing everything unless told otherwise.
    Chorus::Logger logger(std::string source = "test", Chorus::LogLevel minimum = Chorus::LogLevel::Debug) {
        return Chorus::Logger(sink(), minimum, std::move(source));
    }

    std::vector<Chorus::LogRecord> records() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _records;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _records.size();
    }

    // How many records carry `message` verbatim. Verbatim on purpose: a
    // message with an interpolated value in it is the defect this asserts away.
    size_t count(const std::string& message) const {
        std::lock_guard<std::mutex> lock(_mutex);
        size_t seen = 0;
        for (const auto& record : _records)
            seen += record.message == message ? 1 : 0;
        return seen;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _records.clear();
    }

  private:
    mutable std::mutex _mutex;
    std::vector<Chorus::LogRecord> _records;
};

/// The value of `key` on `record`, or nullptr when it carries no such field.
inline const Chorus::LogValue* find_log_field(const Chorus::LogRecord& record, const std::string& key) {
    for (const auto& field : record.fields) {
        if (field.first == key)
            return &field.second;
    }
    return nullptr;
}
