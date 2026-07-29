#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "wlib/utf8.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace Chorus {
namespace {

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
        const size_t safe_length = wlib::valid_utf8_prefix_length(std::string_view(_pending).substr(0, match_position));
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
    size_t remaining = emission_limit;
    std::string safe_text;
    while (remaining > 0) {
        const std::string_view candidate(_pending.data(), remaining);
        const size_t safe_length = wlib::valid_utf8_prefix_length(candidate);
        safe_text.append(_pending, 0, safe_length);
        _pending.erase(0, safe_length);
        remaining -= safe_length;
        if (remaining == 0)
            break;
        const std::string_view unresolved(_pending.data(), remaining);
        if (wlib::is_utf8_incomplete_sequence(unresolved))
            break;
        _pending.erase(0, 1);
        --remaining;
    }
    return {std::move(safe_text), false};
}

StopFilterResult StopSequenceFilter::finish(std::string_view final_piece) {
    StopFilterResult result = push(final_piece);
    if (!result.matched)
        result.safe_text += flush();
    return result;
}

std::string StopSequenceFilter::flush() {
    std::string result;
    if (wlib::valid_utf8_prefix_length(_pending) == _pending.size())
        result = std::move(_pending);
    _pending.clear();
    return result;
}

void StopSequenceFilter::reset() {
    _pending.clear();
}

StopFilterResult finish_content_stream(
    StopSequenceFilter* stop_filter, wlib::Utf8Chunker& content_chunker, std::string_view final_piece
) {
    if (stop_filter)
        return stop_filter->finish(final_piece);
    return {content_chunker.push(final_piece), false};
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
