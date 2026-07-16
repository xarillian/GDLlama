#include "chorus/backends/llama/stop_sequence_filter.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace Chorus {
namespace {

bool is_continuation(unsigned char byte) {
    return byte >= 0x80 && byte <= 0xBF;
}

size_t valid_utf8_prefix_length(std::string_view text) {
    size_t index = 0;
    while (index < text.size()) {
        const auto first = static_cast<unsigned char>(text[index]);
        if (first <= 0x7F) {
            ++index;
            continue;
        }

        size_t width = 0;
        if (first >= 0xC2 && first <= 0xDF)
            width = 2;
        else if (first >= 0xE0 && first <= 0xEF)
            width = 3;
        else if (first >= 0xF0 && first <= 0xF4)
            width = 4;
        else
            break;

        if (text.size() - index < width)
            break;

        const auto second = static_cast<unsigned char>(text[index + 1]);
        if (!is_continuation(second))
            break;
        if ((first == 0xE0 && second < 0xA0) || (first == 0xED && second > 0x9F) || (first == 0xF0 && second < 0x90) ||
            (first == 0xF4 && second > 0x8F))
            break;

        bool valid = true;
        for (size_t offset = 2; offset < width; ++offset) {
            if (!is_continuation(static_cast<unsigned char>(text[index + offset]))) {
                valid = false;
                break;
            }
        }
        if (!valid)
            break;

        index += width;
    }
    return index;
}

bool is_marker_prefix(std::string_view shorter, std::string_view longer) {
    return longer.starts_with(shorter);
}

} // namespace

StopSequenceFilter::StopSequenceFilter(std::vector<std::string> markers) : _markers(std::move(markers)) {}

StopFilterResult StopSequenceFilter::push(std::string_view piece) {
    _pending.append(piece);

    size_t match_position = std::numeric_limits<size_t>::max();
    for (const auto& marker : _markers) {
        const size_t position = _pending.find(marker);
        if (position != std::string::npos)
            match_position = std::min(match_position, position);
    }

    if (match_position != std::numeric_limits<size_t>::max()) {
        const size_t safe_length = valid_utf8_prefix_length(std::string_view(_pending).substr(0, match_position));
        std::string safe_text = _pending.substr(0, safe_length);
        _pending.clear();
        return {std::move(safe_text), true};
    }

    size_t marker_prefix_length = 0;
    for (const auto& marker : _markers) {
        const size_t maximum = std::min(marker.size(), _pending.size());
        for (size_t length = maximum; length > marker_prefix_length; --length) {
            if (_pending.compare(_pending.size() - length, length, marker, 0, length) == 0) {
                marker_prefix_length = length;
                break;
            }
        }
    }

    const size_t emission_limit = _pending.size() - marker_prefix_length;
    const size_t safe_length = valid_utf8_prefix_length(std::string_view(_pending).substr(0, emission_limit));
    std::string safe_text = _pending.substr(0, safe_length);
    _pending.erase(0, safe_length);
    return {std::move(safe_text), false};
}

std::string StopSequenceFilter::flush() {
    std::string result;
    if (valid_utf8_prefix_length(_pending) == _pending.size())
        result = std::move(_pending);
    _pending.clear();
    return result;
}

void StopSequenceFilter::reset() {
    _pending.clear();
}

std::optional<RequestRejection> validate_stop_sequences(const std::vector<std::string>& markers) {
    for (size_t index = 0; index < markers.size(); ++index) {
        if (markers[index].empty()) {
            return RequestRejection{
                ChorusError::InvalidRequest,
                "Stop marker at index " + std::to_string(index) + " must not be empty.",
            };
        }
    }

    for (size_t left = 0; left < markers.size(); ++left) {
        for (size_t right = left + 1; right < markers.size(); ++right) {
            if (markers[left] == markers[right]) {
                return RequestRejection{
                    ChorusError::InvalidRequest,
                    "Stop markers '" + markers[left] + "' and '" + markers[right] + "' are duplicates.",
                };
            }
            if (is_marker_prefix(markers[left], markers[right]) || is_marker_prefix(markers[right], markers[left])) {
                return RequestRejection{
                    ChorusError::InvalidRequest,
                    "Stop markers '" + markers[left] + "' and '" + markers[right] + "' are prefix-related.",
                };
            }
        }
    }

    return std::nullopt;
}

} // namespace Chorus
