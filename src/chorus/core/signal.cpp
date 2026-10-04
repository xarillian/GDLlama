#include "chorus/core/common.hpp"

namespace Chorus {
namespace {

struct TerminalVisitor {
    bool operator()(const ChorusSignal::Token&) const { return false; }
    bool operator()(const ChorusSignal::Completion&) const { return true; }
    bool operator()(const ChorusSignal::Embedding&) const { return true; }
    bool operator()(const ChorusSignal::Error&) const { return true; }
};

} // namespace

bool ChorusSignal::is_terminal() const {
    return std::visit(TerminalVisitor{}, event);
}

} // namespace Chorus
