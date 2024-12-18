#include "CirPrinter.hpp"
#include "cir/CIR.h"

#include <iostream>
#include <spdlog/spdlog.h>

namespace cudalator {
CirPrinter::CirPrinter() : m_indent(0) {}
CirPrinter::~CirPrinter() {}

void CirPrinter::printAst(cir::Ast& ast) {
    spdlog::info("Printing AST");

    printModule(ast, ast.getTopModule());
}

void CirPrinter::printModule(cir::Ast& ast, const cir::Module& module) {
    std::cout << "(module " << module.name() << "\n";

    m_indent++;
    for (auto signal_idx : module.signals()) {
        auto& signal = ast.getNode(signal_idx);

        printSignal(ast, signal);
    }

    for (auto process_idx : module.processes()) {
        auto& process = ast.getNode(process_idx);
        printProcess(ast, process);
    }

    m_indent--;
    std::cout << ")\n";
}

void CirPrinter::printProcess(cir::Ast& ast, const cir::Process& process) {
    printIndent();
    std::cout << "(process " << process.name() << "\n";
    m_indent++;

    printIndent();
    std::cout << "(sensitive";
    for (auto signal_idx : process.sensitivityList()) {
        auto& signal = ast.getNode(signal_idx);
        std::cout << " " << signal.name();
    }
    std::cout << ")\n";

    printIndent();
    std::cout << "(body\n";

    m_indent++;
    auto& statement = ast.getNode(process.statement());
    printStatement(ast, statement);
    m_indent--;

    printIndent();
    std::cout << ")\n";

    m_indent--;
    printIndent();
    std::cout << ")\n";
}

void CirPrinter::printStatement(cir::Ast& ast,
                                const cir::Statement& statement) {
    switch (statement.kind()) {
    case cir::StatementKind::Assignment: {
        printIndent();
        std::cout << "(assign ";
        auto lhs = ast.getNode(statement.lhs());
        auto rhs = ast.getNode(statement.rhs());

        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);

        std::cout << ")\n";

    } break;
    default: {
        printIndent();
        std::cout << "(unhandled statement)\n";
    } break;
    }
}

void CirPrinter::printExpr(cir::Ast& ast, const cir::Expr& expr) {
    switch (expr.kind()) {
    case cir::ExprKind::SignalRef: {
        auto signal = ast.getNode(expr.signal());
        std::cout << "(signal " << signal.name() << ")";
    } break;

    case cir::ExprKind::Constant: {
        // TODO: This has to be handled better
        auto constant = ast.getNode(expr.constant());
        std::cout << "(constant " << constant.value() << ")";
    } break;

    case cir::ExprKind::Addition: {
        auto lhs = ast.getNode(expr.lhs());
        auto rhs = ast.getNode(expr.rhs());
        std::cout << "(add ";
        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);
        std::cout << ")";
    } break;

    case cir::ExprKind::PartSelect: {
        auto signal = ast.getNode(expr.signal());
        auto lhs = ast.getNode(expr.lhs());
        auto rhs = ast.getNode(expr.rhs());

        std::cout << "(select " << signal.name() << " ";
        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);
        std::cout << ")";
    } break;

    case cir::ExprKind::BitSelect: {
        auto signal = ast.getNode(expr.signal());
        auto expr_idx = expr.exprs()[0];
        auto idx = ast.getNode(expr_idx);

        std::cout << "(select_bit " << signal.name() << " ";
        printExpr(ast, idx);
        std::cout << ")";
    } break;

    default: {
        std::cout << "(unhandled)";
    } break;
    }
}

void CirPrinter::printSignal(cir::Ast& ast, const cir::Signal& signal) {
    printIndent();

    std::cout << "(signal " << signal.name();

    switch (signal.kind()) {
    case cir::SignalDirection::Internal: {
        std::cout << " internal";
    } break;
    case cir::SignalDirection::Input: {
        std::cout << " input";
    } break;
    case cir::SignalDirection::Output: {
        std::cout << " output";
    } break;
    case cir::SignalDirection::Inout: {
        std::cout << " output";
    } break;
    }

    auto type_idx = signal.type();

    if (type_idx.isValid()) {
        auto type = ast.getNode(type_idx);
        printType(ast, type);
    }

    std::cout << ")\n";
}

void CirPrinter::printType(cir::Ast& ast, const cir::Type& type) {
    auto kind = type.kind();

    switch (kind) {
    case cir::TypeKind::Logic: {
        std::cout << " logic";
    } break;
    case cir::TypeKind::Bit: {
        std::cout << " bit";
    } break;
    case cir::TypeKind::Integer: {
        std::cout << " integer";
    } break;
    }

    for (auto range : type.ranges()) {
        std::cout << " [" << range.left() << ":" << range.right() << "]";
    }
}

void CirPrinter::printIndent() {
    for (int i = 0; i < m_indent; i++) {
        std::cout << " ";
    }
}

} // namespace cudalator
