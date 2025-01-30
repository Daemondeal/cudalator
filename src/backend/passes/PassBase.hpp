#pragma once

#include "cir/CIR.h"
namespace cudalator {
class PassBase {
public:
    PassBase(cir::Ast& ast) : m_ast(ast) {}

    virtual void runPass() = 0;

protected:
    cir::Ast& m_ast;
};
} // namespace cudalator
