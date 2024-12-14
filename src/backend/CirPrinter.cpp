#include "CirPrinter.hpp"

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
    for (auto signal_idx : process.signals()) {
        auto& signal = ast.getNode(signal_idx);
        std::cout << " " << signal.name();
    }
    std::cout << ")\n";

    m_indent--;
    printIndent();
    std::cout << ")\n";
}

void CirPrinter::printStatement(cir::Ast& ast,
                                const cir::Statement& statement) {}

void CirPrinter::printExpr(cir::Ast& ast, const cir::Expr& expr) {}

void CirPrinter::printSignal(cir::Ast& ast, const cir::Signal& signal) {
    printIndent();

    std::cout << "(signal ";

    switch (signal.kind()) {
    case cir::SignalDirection::Internal: {
        std::cout << "internal ";
    } break;
    case cir::SignalDirection::Input: {
        std::cout << "input ";
    } break;
    case cir::SignalDirection::Output: {
        std::cout << "output ";
    } break;
    case cir::SignalDirection::Inout: {
        std::cout << "output ";
    } break;
    }

    // TODO: Print Type
    std::cout << signal.name() << ")\n";
}

void CirPrinter::printType(cir::Ast& ast, const cir::Type& type) {}

void CirPrinter::printIndent() {
    for (int i = 0; i < m_indent; i++) {
        std::cout << " ";
    }
}

} // namespace cudalator
