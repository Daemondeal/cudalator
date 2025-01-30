#pragma once

#include "cir/CIR.h"
#include <cstdint>
#include <iostream>

namespace cudalator {
class CirPrinter {
public:
    CirPrinter();
    CirPrinter(std::ostream& out);
    ~CirPrinter();

    void printPort(cir::Ast& ast, const cir::ModulePort& port);

    void printAst(cir::Ast& ast);
    void printModule(cir::Ast& ast, const cir::Module& module);
    void printProcess(cir::Ast& ast, const cir::Process& process);
    void printStatement(cir::Ast& ast, const cir::Statement& statement);
    void printExpr(cir::Ast& ast, const cir::Expr& expr);
    void printSignal(cir::Ast& ast, const cir::Signal& signal);
    void printType(cir::Ast& ast, const cir::Type& type);
    void printScope(cir::Ast& ast, const cir::Scope& scope);

    void printUnaryOp(const cir::ExprKind kind);
    void printBinaryOp(const cir::ExprKind kind);

private:
    void printIndent();
    uint32_t m_indent;
    std::ostream& m_out;
};
} // namespace cudalator
