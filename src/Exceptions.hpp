#pragma once

#include "cir/CIR.h"
#include <exception>
#include <string>
namespace cudalator {

class CompilerException : public std::exception {
public:
    CompilerException(std::string message, cir::Loc loc) : m_message(message), m_loc(loc) {}

    const char *what() const noexcept override {
        return m_message.data();
    }

    cir::Loc loc() const noexcept {
        return m_loc;
    }

private:
    std::string m_message;
    cir::Loc m_loc;

};

class UnsupportedException : public CompilerException {
public:
    UnsupportedException(std::string message, cir::Loc loc) : CompilerException(message, loc) {}
};

class UnimplementedException : public CompilerException {
public:
    UnimplementedException(std::string message, cir::Loc loc) : CompilerException(message, loc) {}
};

} // namespace cudalator
