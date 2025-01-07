#pragma once

#include <cstdint>
#include <string_view>

#if __has_include(<fmt/core.h>)
#define _CIR_HAS_FMT_
#include <fmt/core.h>
#endif

namespace cir {

class Loc {
public:
    uint32_t line;
    uint32_t column;
    Loc(uint32_t line, uint32_t col) : line(line), column(col) {}
};

struct NodeBase {
public:
    NodeBase(std::string_view name, Loc loc) : m_name(name), m_loc(loc) {}

    // Removing copy constructor since copying a node outside
    // the ast is always unintended
    NodeBase() = delete;
    NodeBase(const NodeBase& other) = delete;
    NodeBase& operator=(const NodeBase& other) = delete;

    // Need to define the move constructor manually since the
    // copy constructor had to be deleted
    NodeBase(NodeBase&& other) noexcept
        : m_name(other.m_name), m_loc(other.m_loc) {}

    NodeBase& operator=(NodeBase&& other) {
        if (this != &other) {
            m_name = other.m_name;
            m_loc = other.m_loc;
        }
        return *this;
    }

    const std::string_view& name() const {
        return m_name;
    }

    Loc loc() const {
        return m_loc;
    }

private:
    std::string_view m_name;

    Loc m_loc;
};

} // namespace cir

#ifdef _CIR_HAS_FMT_

template <>
struct fmt::formatter<cir::Loc> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const cir::Loc& loc, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "[line: {}, column: {}]", loc.line,
                              loc.column);
    }
};

#endif // _CIR_HAS_FMT_
