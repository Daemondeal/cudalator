#pragma once

#include "Bit.hpp"

enum class ChangeType {
    Posedge,
    Negedge,
    Change,
    NoChange,
};

template <>
struct fmt::formatter<ChangeType> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

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
static HOST_DEVICE ChangeType change_calculate(Bit<N> before, Bit<N> after) {
    if constexpr (N == 1) {
        if (before == Bit<1>(0) && after == Bit<1>(1)) {
            return ChangeType::Posedge;
        } else if (before == Bit<1>(1) && after == Bit<1>(0)) {
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

HOST_DEVICE static ChangeType change_combine(ChangeType a, ChangeType b) {
    switch (a) {
    case ChangeType::NoChange:
        return b;
    case ChangeType::Change:
        return ChangeType::Change;
    case ChangeType::Posedge:
        return (b == ChangeType::Posedge) ? ChangeType::Posedge : ChangeType::Change;
    case ChangeType::Negedge:
        return (b == ChangeType::Negedge) ? ChangeType::Negedge : ChangeType::Change;
    }
    return ChangeType::NoChange;
}
