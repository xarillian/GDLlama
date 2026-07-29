#pragma once

#include "chorus/core/common.hpp"
#include "wlib/utf8.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Chorus {

struct StopFilterResult {
    std::string safe_text;
    bool matched = false;
};

class StopSequenceFilter {
  public:
    explicit StopSequenceFilter(std::vector<std::string> markers);

    StopFilterResult push(std::string_view piece);
    StopFilterResult finish(std::string_view final_piece);
    std::string flush();
    void reset();

  private:
    std::vector<std::string> _markers;
    std::string _pending;
};

StopFilterResult finish_content_stream(
    StopSequenceFilter* stop_filter, wlib::Utf8Chunker& content_chunker, std::string_view final_piece
);

std::optional<RequestRejection> validate_stop_sequences(const std::vector<std::string>& markers);

} // namespace Chorus
