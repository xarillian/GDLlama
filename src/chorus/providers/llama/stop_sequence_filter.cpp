#include "chorus/providers/llama/stop_sequence_filter.hpp"
#include "wlib/utf8.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>

namespace Chorus {
namespace {

std::optional<size_t> find_earliest_marker(std::string_view pending, const std::vector<std::string>& markers) {
    std::optional<size_t> earliest;
    for (const auto& marker : markers) {
        const size_t position = pending.find(marker);
        if (position != std::string_view::npos && (!earliest || position < *earliest))
            earliest = position;
    }
    return earliest;
}

size_t find_pending_marker_prefix_length(std::string_view pending, const std::vector<std::string>& markers) {
    size_t longest_prefix = 0;
    for (const auto& marker : markers) {
        const size_t maximum = std::min(marker.size(), pending.size());
        for (size_t length = maximum; length > longest_prefix; --length) {
            if (pending.compare(pending.size() - length, length, marker, 0, length) == 0) {
                longest_prefix = length;
                break;
            }
        }
    }
    return longest_prefix;
}

std::string release_valid_text(std::string& pending, size_t emission_limit) {
    size_t consumed = 0;
    std::string safe_text;
    safe_text.reserve(emission_limit);

    while (consumed < emission_limit) {
        const std::string_view candidate(pending.data() + consumed, emission_limit - consumed);
        const size_t safe_length = wlib::valid_utf8_prefix_length(candidate);
        safe_text.append(pending, consumed, safe_length);
        consumed += safe_length;
        if (consumed == emission_limit)
            break;

        const std::string_view unresolved(pending.data() + consumed, emission_limit - consumed);
        if (wlib::is_utf8_incomplete_sequence(unresolved))
            break;

        // Reject one malformed byte without joining marker matching across it.
        ++consumed;
    }

    pending.erase(0, consumed);
    return safe_text;
}

bool is_marker_prefix(std::string_view shorter, std::string_view longer) {
    return longer.starts_with(shorter);
}

} // namespace

StopSequenceFilter::StopSequenceFilter(std::vector<std::string> markers) : _markers(std::move(markers)) {}

StopFilterResult StopSequenceFilter::push(std::string_view piece) {
    _pending.append(piece);

    if (const auto match_position = find_earliest_marker(_pending, _markers)) {
        const size_t safe_length =
            wlib::valid_utf8_prefix_length(std::string_view(_pending).substr(0, *match_position));
        std::string safe_text = _pending.substr(0, safe_length);
        _pending.clear();
        return {std::move(safe_text), true};
    }

    const size_t held_marker_prefix = find_pending_marker_prefix_length(_pending, _markers);
    const size_t emission_limit = _pending.size() - held_marker_prefix;
    return {release_valid_text(_pending, emission_limit), false};
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
