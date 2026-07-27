#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace wlib {
namespace detail {

inline constexpr bool is_utf8_continuation(unsigned char byte) noexcept {
    return byte >= 0x80 && byte <= 0xBF;
}

// Total sequence width implied by a lead byte; 0 => not a valid lead.
inline constexpr std::size_t utf8_sequence_width(unsigned char lead) noexcept {
    if (lead <= 0x7F)
        return 1;
    if (lead >= 0xC2 && lead <= 0xDF)
        return 2;
    if (lead >= 0xE0 && lead <= 0xEF)
        return 3;
    if (lead >= 0xF0 && lead <= 0xF4)
        return 4;
    return 0;
}

// Continuation check plus the range restrictions that exclude overlong
// encodings and non-scalar values.
inline constexpr bool is_valid_utf8_second_byte(unsigned char lead, unsigned char second) noexcept {
    if (!is_utf8_continuation(second))
        return false;
    if ((lead == 0xE0 && second < 0xA0) || (lead == 0xED && second > 0x9F) || (lead == 0xF0 && second < 0x90) ||
        (lead == 0xF4 && second > 0x8F))
        return false;
    return true;
}

} // namespace detail

inline std::size_t valid_utf8_prefix_length(std::string_view text) noexcept {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto first = static_cast<unsigned char>(text[index]);
        const std::size_t width = detail::utf8_sequence_width(first);
        if (width == 0)
            break;
        if (width == 1) {
            ++index;
            continue;
        }
        if (text.size() - index < width)
            break;
        if (!detail::is_valid_utf8_second_byte(first, static_cast<unsigned char>(text[index + 1])))
            break;

        bool valid = true;
        for (std::size_t offset = 2; offset < width; ++offset) {
            if (!detail::is_utf8_continuation(static_cast<unsigned char>(text[index + offset]))) {
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

// True when `text` is exactly the beginning of one multi-byte scalar that is
// missing trailing bytes and could still complete.
inline constexpr bool is_utf8_incomplete_sequence(std::string_view text) noexcept {
    if (text.empty())
        return false;
    const auto lead = static_cast<unsigned char>(text[0]);
    const std::size_t width = detail::utf8_sequence_width(lead);
    if (width < 2 || text.size() >= width)
        return false;
    if (text.size() >= 2 && !detail::is_valid_utf8_second_byte(lead, static_cast<unsigned char>(text[1])))
        return false;
    for (std::size_t index = 2; index < text.size(); ++index) {
        if (!detail::is_utf8_continuation(static_cast<unsigned char>(text[index])))
            return false;
    }
    return true;
}

// What to do with a rejected byte (one that can no longer begin or continue
// a scalar). Drop excises it silently: a stream guard should not invent
// bytes the source never produced, and downstream text reads cleaner without
// replacement glyphs. Replace substitutes one U+FFFD per rejected byte (the
// conventional lossy-decode behavior) for consumers that need corruption to
// stay visible. Per-byte, deliberately: no WHATWG maximal-subsequence
// coalescing.
enum class Utf8InvalidBytePolicy { Drop, Replace };

// Re-chunks a byte stream on UTF-8 scalar boundaries: push() releases the
// longest valid prefix of what has accumulated, holds back a tail that is
// merely incomplete, and rejects bytes that can no longer form a scalar per
// the policy above (one malformed byte must never dam the stream). A tail
// still held at end of stream is unfinishable by definition; reset()
// discards it without replacement under either policy (reset has no output).
class Utf8Chunker {
  public:
    explicit Utf8Chunker(Utf8InvalidBytePolicy policy = Utf8InvalidBytePolicy::Drop) : _policy(policy) {}

    std::string push(std::string_view piece) {
        _pending.append(piece);
        std::string released;
        while (!_pending.empty()) {
            const std::size_t valid = valid_utf8_prefix_length(_pending);
            released.append(_pending, 0, valid);
            _pending.erase(0, valid);
            if (_pending.empty() || is_utf8_incomplete_sequence(_pending))
                break;
            if (_policy == Utf8InvalidBytePolicy::Replace)
                released.append("\xEF\xBF\xBD"); // U+FFFD replacement character
            _pending.erase(0, 1);                // reject the malformed byte and rescan
        }
        return released;
    }

    void reset() { _pending.clear(); }

  private:
    Utf8InvalidBytePolicy _policy;
    std::string _pending;
};

} // namespace wlib
