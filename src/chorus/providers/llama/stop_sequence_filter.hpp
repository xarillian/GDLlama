#pragma once

#include "chorus/core/common.hpp"
#include "wlib/utf8.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Chorus {

/*
 * Carries text safe for immediate emission and stop-marker status.
 *
 * `Chorus::StopFilterResult::safe_text` contains complete, valid UTF-8 before
 * the earliest matched marker. `Chorus::StopFilterResult::matched` is `true`
 * when any configured marker was consumed; the marker and later bytes are
 * excluded from the result.
 */
struct StopFilterResult {
    std::string safe_text;
    bool matched = false;
};

/*
 * Removes configured byte markers from incrementally generated UTF-8 text.
 *
 * The filter retains only a suffix that may still complete a marker or UTF-8
 * code point. Every earlier complete code point is released immediately.
 */
class StopSequenceFilter {
  public:
    explicit StopSequenceFilter(std::vector<std::string> markers);

    /*
     * Accepts the next generated piece and releases text known not to contain
     * a configured marker.
     */
    StopFilterResult push(std::string_view piece);

    /*
     * Accepts the final piece and releases a pending partial marker when no
     * complete marker matches.
     */
    StopFilterResult finish(std::string_view final_piece);

    /*
     * Clears the filter, releasing pending text only when it is entirely
     * valid UTF-8.
     */
    std::string flush();

  private:
    std::vector<std::string> _markers;
    std::string _pending;
};

/*
 * Finalizes streamed content through its configured UTF-8 path.
 *
 * A non-null `stop_filter` performs marker matching and releases pending
 * partial-marker text. Without one, `final_piece` passes through
 * `content_chunker`, which leaves an incomplete UTF-8 tail un-emitted.
 */
StopFilterResult finish_content_stream(
    StopSequenceFilter* stop_filter, wlib::Utf8Chunker& content_chunker, std::string_view final_piece
);

/*
 * Validates the marker constraints required by streaming stop detection.
 *
 * Failure is returned as a request rejection.
 *
 * Returns:
 *  - `std::nullopt`: markers are non-empty, unique, and no marker prefixes another.
 *  - `Chorus::RequestRejection`: a marker constraint is violated.
 *
 * Errors:
 *  - `Chorus::ChorusError::InvalidRequest`: a marker is empty, duplicated, or prefix-related.
 */
std::optional<RequestRejection> validate_stop_sequences(const std::vector<std::string>& markers);

} // namespace Chorus
