#pragma once

#include "cir/CIR.h"
#include <string>
namespace cudalator {

enum class FrontendErrorKind { Todo, Unsupported, Other };

class FrontendError {
public:
    static inline FrontendError todo(std::string message, cir::Loc loc) {
        return FrontendError(message, loc, FrontendErrorKind::Todo);
    }

    static inline FrontendError unsupported(std::string message, cir::Loc loc) {
        return FrontendError(message, loc, FrontendErrorKind::Unsupported);
    }

    static inline FrontendError other(std::string message, cir::Loc loc) {
        return FrontendError(message, loc, FrontendErrorKind::Other);
    }

    const std::string message() const {
        return m_message;
    }

    cir::Loc loc() const {
        return m_loc;
    }

    FrontendErrorKind kind() const {
        return m_kind;
    }

private:
    FrontendError(std::string message, cir::Loc loc, FrontendErrorKind kind)
        : m_message(message), m_loc(loc), m_kind(kind) {}

    std::string m_message;
    cir::Loc m_loc;
    FrontendErrorKind m_kind;
};
} // namespace cudalator
