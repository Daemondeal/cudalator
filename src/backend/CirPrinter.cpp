#include "CirPrinter.hpp"
#include "cir/CIR.h"

#include <iostream>
#include <spdlog/spdlog.h>
#include <string_view>

namespace cudalator {
CirPrinter::CirPrinter() : m_indent(0) {}
CirPrinter::~CirPrinter() {}

void CirPrinter::printAst(cir::Ast& ast) {
    spdlog::info("Printing AST");

    printModule(ast, ast.getTopModule());
}

void CirPrinter::printPort(cir::Ast& ast, const cir::ModulePort& port) {
    auto& ast_signal = ast.getNode(port.signal);

    printIndent();
    std::cout << "(port " << ast_signal.name();

    switch (port.direction) {
    case cir::SignalDirection::Input: {
        std::cout << " input";
    } break;
    case cir::SignalDirection::Output: {
        std::cout << " output";
    } break;
    case cir::SignalDirection::Inout: {
        std::cout << " inout";
    } break;
    case cir::SignalDirection::Invalid: {
        std::cout << " invalid";
    } break;
    }

    std::cout << ")\n";
}

void CirPrinter::printModule(cir::Ast& ast, const cir::Module& module) {
    std::cout << "(module " << module.name() << "\n";

    m_indent++;
    printIndent();
    std::cout << "(ports\n";

    m_indent++;
    for (auto port : module.ports()) {
        printPort(ast, port);
    }

    printIndent();
    std::cout << ")\n";
    m_indent--;

    m_indent++;
    printScope(ast, ast.getNode(module.scope()));
    m_indent--;

    for (auto process_idx : module.processes()) {
        auto& process = ast.getNode(process_idx);
        printProcess(ast, process);
    }

    m_indent--;
    std::cout << ")\n";
}

void CirPrinter::printScope(cir::Ast& ast, const cir::Scope& scope) {
    for (auto signal_idx : scope.signals()) {
        auto& signal = ast.getNode(signal_idx);

        printSignal(ast, signal);
    }
}

void CirPrinter::printProcess(cir::Ast& ast, const cir::Process& process) {
    printIndent();
    std::cout << "(process " << process.name() << "\n";
    m_indent++;
    printIndent();
    std::cout << "(sensitive";
    for (auto element : process.sensitivityList()) {
        if (element.kind == cir::SensitivityKind::Negedge) {
            std::cout << " negedge";
        } else if (element.kind == cir::SensitivityKind::Posedge) {
            std::cout << " posedge";
        }

        auto& signal = ast.getNode(element.signal);
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
        auto& lhs = ast.getNode(statement.lhs());
        auto& rhs = ast.getNode(statement.rhs());

        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);

        std::cout << ")\n";

    } break;
    case cir::StatementKind::NonBlockingAssignment: {
        printIndent();
        std::cout << "(assign_nonblock ";
        auto& lhs = ast.getNode(statement.lhs());
        auto& rhs = ast.getNode(statement.rhs());

        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);

        std::cout << ")\n";

    } break;
    case cir::StatementKind::Block: {
        printIndent();
        std::cout << "(block \n";

        m_indent++;
        if (statement.scope().isValid()) {
            auto& scope = ast.getNode(statement.scope());
            printScope(ast, scope);
        }

        for (auto sub_idx : statement.statements()) {
            auto& sub_stmt = ast.getNode(sub_idx);
            printStatement(ast, sub_stmt);
        }
        m_indent--;

        printIndent();
        std::cout << ")\n";

    } break;
    case cir::StatementKind::If: {
        printIndent();
        std::cout << "(if ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        std::cout << "\n";

        auto& body = ast.getNode(statement.body());
        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        std::cout << ")\n";

    } break;
    case cir::StatementKind::IfElse: {
        printIndent();
        std::cout << "(if ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        std::cout << "\n";

        auto& body = ast.getNode(statement.statement(0));
        auto& else_stmt = ast.getNode(statement.statement(1));

        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        std::cout << "else (\n";

        m_indent++;
        printStatement(ast, else_stmt);
        m_indent--;

        printIndent();
        std::cout << ")\n";

    } break;
    case cir::StatementKind::While: {
        printIndent();
        std::cout << "(while ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        std::cout << "\n";

        auto& body = ast.getNode(statement.body());

        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        std::cout << ")\n";
    } break;
    case cir::StatementKind::DoWhile: {
        printIndent();
        std::cout << "(do\n";

        auto& body = ast.getNode(statement.body());

        m_indent++;
        printStatement(ast, body);
        m_indent--;

        printIndent();
        std::cout << "while ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        std::cout << ")\n";
    } break;
    case cir::StatementKind::Repeat: {
        printIndent();
        std::cout << "(repeat ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        std::cout << "\n";

        auto& body = ast.getNode(statement.body());

        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        std::cout << ")\n";
    } break;
    case cir::StatementKind::For: {
        printIndent();
        std::cout << "(for\n";

        m_indent++;
        auto& init = ast.getNode(statement.statement(0));
        printStatement(ast, init);

        printIndent();
        printExpr(ast, ast.getNode(statement.lhs()));

        auto& incr = ast.getNode(statement.statement(1));
        printStatement(ast, incr);

        auto& body = ast.getNode(statement.statement(2));
        printStatement(ast, body);
        m_indent--;

        printIndent();
        std::cout << ")\n";
    } break;

    case cir::StatementKind::Forever: {
        printIndent();
        std::cout << "(forever\n";

        m_indent++;
        auto& body = ast.getNode(statement.body());
        printStatement(ast, body);
        m_indent--;

        printIndent();
        std::cout << ")\n";

    } break;

    case cir::StatementKind::Return: {
        printIndent();
        std::cout << "(return ";
        auto& rval = ast.getNode(statement.lhs());
        printExpr(ast, rval);
        std::cout << ")\n";
    } break;

    case cir::StatementKind::Break: {
        printIndent();
        std::cout << "(break)\n";
    } break;

    case cir::StatementKind::Continue: {
        printIndent();
        std::cout << "(continue)\n";
    } break;

    case cir::StatementKind::Null: {
        printIndent();
        std::cout << "(null)\n";
    } break;

    default: {
        printIndent();
        std::cout << "(unhandled statement)\n";
    } break;
    }
}

void CirPrinter::printExpr(cir::Ast& ast, const cir::Expr& expr) {
    if (expr.isUnary()) {
        std::cout << "(";
        printUnaryOp(expr.kind());

        std::cout << " ";
        auto& operand = ast.getNode(expr.expr(0));
        printExpr(ast, operand);
        std::cout << ")";
        return;
    }

    if (expr.isBinary()) {
        std::cout << "(";
        printBinaryOp(expr.kind());

        std::cout << " ";
        auto& lhs = ast.getNode(expr.lhs());
        auto& rhs = ast.getNode(expr.rhs());

        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);
        std::cout << ")";
        return;
    }

    switch (expr.kind()) {
    case cir::ExprKind::SignalRef: {
        std::string_view name;
        if (expr.signal().isValid()) {
            auto& signal = ast.getNode(expr.signal());
            name = signal.name();
        } else {
            name = "invalid";
        }

        std::cout << "(signal " << name << ")";
    } break;

    case cir::ExprKind::Constant: {
        // TODO: This has to be handled better
        auto& constant = ast.getNode(expr.constant());
        std::cout << "(constant " << constant.value() << ")";
    } break;

    case cir::ExprKind::PartSelect: {
        auto& signal = ast.getNode(expr.signal());
        auto& lhs = ast.getNode(expr.lhs());
        auto& rhs = ast.getNode(expr.rhs());

        std::cout << "(select " << signal.name() << " ";
        printExpr(ast, lhs);
        std::cout << " ";
        printExpr(ast, rhs);
        std::cout << ")";
    } break;

    case cir::ExprKind::BitSelect: {
        auto& signal = ast.getNode(expr.signal());
        auto& expr_idx = expr.exprs()[0];
        auto& idx = ast.getNode(expr_idx);

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

    switch (signal.lifetime()) {
    case cir::SignalLifetime::Static: {
        std::cout << " static";
    } break;
    case cir::SignalLifetime::Automatic: {
        std::cout << " automatic";
    } break;
    case cir::SignalLifetime::Net: {
        std::cout << " net";
    } break;
    }

    auto type_idx = signal.type();

    if (type_idx.isValid()) {
        auto& type = ast.getNode(type_idx);
        printType(ast, type);
    }

    std::cout << ")\n";
}

void CirPrinter::printType(cir::Ast& ast, const cir::Type& type) {
    (void)ast;

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
    case cir::TypeKind::Int: {
        std::cout << " int";
    } break;
    case cir::TypeKind::Invalid: {
        std::cout << " invalid";
    } break;
    }

    for (auto range : type.ranges()) {
        std::cout << " [" << range.left() << ":" << range.right() << "]";
    }
}

void CirPrinter::printUnaryOp(const cir::ExprKind kind) {
    switch (kind) {
    case cir::ExprKind::UnaryMinus:
        std::cout << "-";
        break;
    case cir::ExprKind::UnaryPlus:
        std::cout << "+";
        break;
    case cir::ExprKind::Not:
        std::cout << "!";
        break;
    case cir::ExprKind::BinaryNegation:
        std::cout << "~";
        break;
    case cir::ExprKind::ReductionAnd:
        std::cout << "red_and";
        break;
    case cir::ExprKind::ReductionNand:
        std::cout << "red_nand";
        break;
    case cir::ExprKind::ReductionOr:
        std::cout << "red_or";
        break;
    case cir::ExprKind::ReductionNor:
        std::cout << "red_nor";
        break;
    case cir::ExprKind::ReductionXor:
        std::cout << "red_xor";
        break;
    case cir::ExprKind::ReductionXnor:
        std::cout << "red_xnor";
        break;
    case cir::ExprKind::Posedge:
        std::cout << "posedge";
        break;
    case cir::ExprKind::Negedge:
        std::cout << "negedge";
        break;
    default:
        std::cout << "invalid_unary";
        break;
    }
}

void CirPrinter::printBinaryOp(const cir::ExprKind kind) {
    switch (kind) {
    case cir::ExprKind::Subtraction:
        std::cout << "-";
        break;
    case cir::ExprKind::Division:
        std::cout << "/";
        break;
    case cir::ExprKind::Modulo:
        std::cout << "%";
        break;
    case cir::ExprKind::Equality:
        std::cout << "==";
        break;
    case cir::ExprKind::NotEquality:
        std::cout << "!=";
        break;
    case cir::ExprKind::GreaterThan:
        std::cout << ">";
        break;
    case cir::ExprKind::GreaterThanEq:
        std::cout << ">=";
        break;
    case cir::ExprKind::LessThan:
        std::cout << "<";
        break;
    case cir::ExprKind::LessThanEq:
        std::cout << "<=";
        break;
    case cir::ExprKind::LeftShift:
        std::cout << "<<";
        break;
    case cir::ExprKind::RightShift:
        std::cout << ">>";
        break;
    case cir::ExprKind::Addition:
        std::cout << "+";
        break;
    case cir::ExprKind::Multiplication:
        std::cout << "*";
        break;
    case cir::ExprKind::LogicalAnd:
        std::cout << "and";
        break;
    case cir::ExprKind::LogicalOr:
        std::cout << "or";
        break;
    case cir::ExprKind::BitwiseAnd:
        std::cout << "&";
        break;
    case cir::ExprKind::BitwiseOr:
        std::cout << "|";
        break;
    case cir::ExprKind::BitwiseXor:
        std::cout << "^";
        break;
    case cir::ExprKind::BitwiseXnor:
        std::cout << "~^";
        break;
    default:
        std::cout << "binary_invalid";
    }
}

void CirPrinter::printIndent() {
    for (size_t i = 0; i < m_indent; i++) {
        std::cout << " ";
    }
}

} // namespace cudalator
