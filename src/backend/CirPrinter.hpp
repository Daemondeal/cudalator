#pragma once

#include "cir/CIR.h"
#include <cstdint>

namespace cudalator {
class CirPrinter {
public:
    CirPrinter();
    ~CirPrinter();

    void printAst(cir::Ast& ast);
    void printModule(cir::Ast& ast, const cir::Module& module);
    void printProcess(cir::Ast& ast, const cir::Process& process);
    void printStatement(cir::Ast& ast, const cir::Statement& statement);
    void printExpr(cir::Ast& ast, const cir::Expr& expr);
    void printSignal(cir::Ast& ast, const cir::Signal& signal);
    void printType(cir::Ast& ast, const cir::Type& type);

private:

    void printIndent();
    uint32_t m_indent;
};
} // namespace cudalator
