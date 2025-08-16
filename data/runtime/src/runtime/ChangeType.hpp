#pragma once

#include "Bit.hpp"

enum class ChangeType {
    Posedge,
    Negedge,
    Change,
    NoChange,
};

template<>
struct fmt::formatter<ChangeType> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const ChangeType& n, FormatContext& ctx) const {
        switch (n) {
        case ChangeType::Posedge:
            return fmt::format_to(ctx.out(), "Posedge");
        case ChangeType::Negedge:
            return fmt::format_to(ctx.out(), "Negedge");
        case ChangeType::Change:
            return fmt::format_to(ctx.out(), "Change");
        case ChangeType::NoChange:
            return fmt::format_to(ctx.out(), "NoChange");
          break;
        }
        return fmt::format_to(ctx.out(), "Invalid");
    }
};

template <int N>
static ChangeType change_calculate(Bit<N> before, Bit<N> after) {
    if constexpr (N == 1) {
        if (before == 0 && after == 1) {
            return ChangeType::Posedge;
        } else if (before == 1 && after == 0) {
            return ChangeType::Negedge;
        }
        return ChangeType::NoChange;
    }

    if (before == after) {
        return ChangeType::NoChange;
    } else {
        return ChangeType::Change;
    }
};

