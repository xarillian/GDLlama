#pragma once

#include "chorus/core/capabilities.hpp"

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
    std::string flush();
    void reset();

  private:
    std::vector<std::string> _markers;
    std::string _pending;
};

std::optional<RequestRejection> validate_stop_sequences(const std::vector<std::string>& markers);

} // namespace Chorus
