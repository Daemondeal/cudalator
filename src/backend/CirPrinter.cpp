#include "CirPrinter.hpp"
#include "cir/CIR.h"

#include <iostream>
#include <spdlog/spdlog.h>
#include <string_view>

namespace cudalator {
CirPrinter::CirPrinter() : m_indent(0), m_out(std::cout) {}
CirPrinter::CirPrinter(std::ostream& out) : m_indent(0), m_out(out) {}
CirPrinter::~CirPrinter() {}

void CirPrinter::printAst(cir::Ast& ast) {
    spdlog::info("Printing AST");

    printModule(ast, ast.getTopModule());
}

void CirPrinter::printPort(cir::Ast& ast, const cir::ModulePort& port) {
    auto& ast_signal = ast.getNode(port.signal);

    printIndent();
    m_out << "(port " << ast_signal.name();

    switch (port.direction) {
    case cir::SignalDirection::Input: {
        m_out << " input";
    } break;
    case cir::SignalDirection::Output: {
        m_out << " output";
    } break;
    case cir::SignalDirection::Inout: {
        m_out << " inout";
    } break;
    case cir::SignalDirection::Invalid: {
        m_out << " invalid";
    } break;
    }

    m_out << ")\n";
}

void CirPrinter::printModule(cir::Ast& ast, const cir::Module& module) {
    m_out << "(module " << module.name() << "\n";

    m_indent++;
    printIndent();
    m_out << "(ports";

    if (module.ports().size() > 0) {
        m_out << "\n";

        m_indent++;
        for (auto port : module.ports()) {
            printPort(ast, port);
        }

        printIndent();
        m_indent--;
    }
    m_out << ")\n";

    m_indent++;
    printScope(ast, ast.getNode(module.scope()));
    m_indent--;

    for (auto process_idx : module.processes()) {
        auto& process = ast.getNode(process_idx);
        printProcess(ast, process);
    }

    m_indent--;
    m_out << ")\n";
}

void CirPrinter::printScope(cir::Ast& ast, const cir::Scope& scope) {
    for (auto signal_idx : scope.signals()) {
        auto& signal = ast.getNode(signal_idx);

        printSignal(ast, signal);
    }
}

void CirPrinter::printProcess(cir::Ast& ast, const cir::Process& process) {
    printIndent();
    m_out << "(process " << process.name() << "\n";
    m_indent++;
    printIndent();
    m_out << "(sensitive";
    for (auto element : process.sensitivityList()) {
        if (element.kind == cir::SensitivityKind::Negedge) {
            m_out << " negedge";
        } else if (element.kind == cir::SensitivityKind::Posedge) {
            m_out << " posedge";
        }

        auto& signal = ast.getNode(element.signal);
        m_out << " " << signal.name();
    }
    m_out << ")\n";

    printIndent();
    m_out << "(body\n";

    m_indent++;
    auto& statement = ast.getNode(process.statement());
    printStatement(ast, statement);
    m_indent--;

    printIndent();
    m_out << ")\n";

    m_indent--;
    printIndent();
    m_out << ")\n";
}

void CirPrinter::printStatement(cir::Ast& ast,
                                const cir::Statement& statement) {
    switch (statement.kind()) {
    case cir::StatementKind::Assignment: {
        printIndent();
        m_out << "(assign ";
        auto& lhs = ast.getNode(statement.lhs());
        auto& rhs = ast.getNode(statement.rhs());

        printExpr(ast, lhs);
        m_out << " ";
        printExpr(ast, rhs);

        m_out << ")\n";

    } break;
    case cir::StatementKind::NonBlockingAssignment: {
        printIndent();
        m_out << "(assign_nonblock ";
        auto& lhs = ast.getNode(statement.lhs());
        auto& rhs = ast.getNode(statement.rhs());

        printExpr(ast, lhs);
        m_out << " ";
        printExpr(ast, rhs);

        m_out << ")\n";

    } break;
    case cir::StatementKind::Block: {
        printIndent();
        m_out << "(block \n";

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
        m_out << ")\n";

    } break;
    case cir::StatementKind::If: {
        printIndent();
        m_out << "(if ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        m_out << "\n";

        auto& body = ast.getNode(statement.body());
        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        m_out << ")\n";

    } break;
    case cir::StatementKind::IfElse: {
        printIndent();
        m_out << "(if ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        m_out << "\n";

        auto& body = ast.getNode(statement.statement(0));
        auto& else_stmt = ast.getNode(statement.statement(1));

        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        m_out << "else (\n";

        m_indent++;
        printStatement(ast, else_stmt);
        m_indent--;

        printIndent();
        m_out << ")\n";

    } break;
    case cir::StatementKind::While: {
        printIndent();
        m_out << "(while ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        m_out << "\n";

        auto& body = ast.getNode(statement.body());

        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        m_out << ")\n";
    } break;
    case cir::StatementKind::DoWhile: {
        printIndent();
        m_out << "(do\n";

        auto& body = ast.getNode(statement.body());

        m_indent++;
        printStatement(ast, body);
        m_indent--;

        printIndent();
        m_out << "while ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        m_out << ")\n";
    } break;
    case cir::StatementKind::Repeat: {
        printIndent();
        m_out << "(repeat ";
        auto& cond = ast.getNode(statement.lhs());
        printExpr(ast, cond);
        m_out << "\n";

        auto& body = ast.getNode(statement.body());

        m_indent++;
        printStatement(ast, body);
        m_indent--;
        printIndent();
        m_out << ")\n";
    } break;
    case cir::StatementKind::For: {
        printIndent();
        m_out << "(for\n";

        m_indent++;

        if (statement.scope().isValid()) {
            auto& scope = ast.getNode(statement.scope());
            if (scope.signals().size() > 0) {
                printIndent();
                m_out << "(scope\n";
                m_indent++;
                printScope(ast, scope);
                m_indent--;
                printIndent();
                m_out << ")\n";
            }
        }

        auto& init = ast.getNode(statement.statement(0));
        printStatement(ast, init);

        printIndent();
        printExpr(ast, ast.getNode(statement.lhs()));
        m_out << "\n";

        auto& incr = ast.getNode(statement.statement(1));
        printStatement(ast, incr);

        auto& body = ast.getNode(statement.statement(2));
        printStatement(ast, body);
        m_indent--;

        printIndent();
        m_out << ")\n";
    } break;

    case cir::StatementKind::Forever: {
        printIndent();
        m_out << "(forever\n";

        m_indent++;
        auto& body = ast.getNode(statement.body());
        printStatement(ast, body);
        m_indent--;

        printIndent();
        m_out << ")\n";

    } break;

    case cir::StatementKind::Return: {
        printIndent();
        m_out << "(return ";
        auto& rval = ast.getNode(statement.lhs());
        printExpr(ast, rval);
        m_out << ")\n";
    } break;

    case cir::StatementKind::Break: {
        printIndent();
        m_out << "(break)\n";
    } break;

    case cir::StatementKind::Continue: {
        printIndent();
        m_out << "(continue)\n";
    } break;

    case cir::StatementKind::Null: {
        printIndent();
        m_out << "(null)\n";
    } break;

    default: {
        printIndent();
        m_out << "(unhandled statement)\n";
    } break;
    }
}

void CirPrinter::printExpr(cir::Ast& ast, const cir::Expr& expr) {
    if (expr.isUnary()) {
        m_out << "(";
        printUnaryOp(expr.kind());

        m_out << " ";
        auto& operand = ast.getNode(expr.expr(0));
        printExpr(ast, operand);
        m_out << ")";
        return;
    }

    if (expr.isBinary()) {
        m_out << "(";
        printBinaryOp(expr.kind());

        m_out << " ";
        auto& lhs = ast.getNode(expr.lhs());
        auto& rhs = ast.getNode(expr.rhs());

        printExpr(ast, lhs);
        m_out << " ";
        printExpr(ast, rhs);
        m_out << ")";
        return;
    }

    switch (expr.kind()) {
    case cir::ExprKind::Concatenation: {
        m_out << "(concat";
        for (auto expr : expr.exprs()) {
            auto& sub_expr = ast.getNode(expr);
            m_out << " ";
            printExpr(ast, sub_expr);
        }
        m_out << ")";

    } break;
    case cir::ExprKind::SignalRef: {
        std::string_view name;
        if (expr.signal().isValid()) {
            auto& signal = ast.getNode(expr.signal());
            name = signal.name();
        } else {
            name = "invalid";
        }

        m_out << "(signal " << name << ")";
    } break;

    case cir::ExprKind::Constant: {
        // TODO: This has to be handled better
        auto& constant = ast.getNode(expr.constant());
        m_out << "(constant " << constant.value() << ")";
    } break;

    case cir::ExprKind::PartSelect: {
        auto& signal = ast.getNode(expr.signal());
        auto& lhs = ast.getNode(expr.lhs());
        auto& rhs = ast.getNode(expr.rhs());

        m_out << "(select " << signal.name() << " ";
        printExpr(ast, lhs);
        m_out << " ";
        printExpr(ast, rhs);
        m_out << ")";
    } break;

    case cir::ExprKind::BitSelect: {
        auto& signal = ast.getNode(expr.signal());
        auto& expr_idx = expr.exprs()[0];
        auto& idx = ast.getNode(expr_idx);

        m_out << "(select_bit " << signal.name() << " ";
        printExpr(ast, idx);
        m_out << ")";
    } break;

    default: {
        m_out << "(unhandled)";
    } break;
    }
}

void CirPrinter::printSignal(cir::Ast& ast, const cir::Signal& signal) {
    printIndent();

    m_out << "(signal " << signal.name();

    switch (signal.lifetime()) {
    case cir::SignalLifetime::Static: {
        m_out << " static";
    } break;
    case cir::SignalLifetime::Automatic: {
        m_out << " automatic";
    } break;
    case cir::SignalLifetime::Net: {
        m_out << " net";
    } break;
    }

    auto type_idx = signal.type();

    if (type_idx.isValid()) {
        auto& type = ast.getNode(type_idx);
        printType(ast, type);
    }

    m_out << ")\n";
}

void CirPrinter::printType(cir::Ast& ast, const cir::Type& type) {
    (void)ast;

    auto kind = type.kind();

    switch (kind) {
    case cir::TypeKind::Logic: {
        m_out << " logic";
    } break;
    case cir::TypeKind::Bit: {
        m_out << " bit";
    } break;
    case cir::TypeKind::Integer: {
        m_out << " integer";
    } break;
    case cir::TypeKind::Int: {
        m_out << " int";
    } break;
    case cir::TypeKind::Invalid: {
        m_out << " invalid";
    } break;
    }

    for (auto range : type.ranges()) {
        m_out << " [" << range.left() << ":" << range.right() << "]";
    }
}

void CirPrinter::printUnaryOp(const cir::ExprKind kind) {
    switch (kind) {
    case cir::ExprKind::UnaryMinus:
        m_out << "-";
        break;
    case cir::ExprKind::UnaryPlus:
        m_out << "+";
        break;
    case cir::ExprKind::Not:
        m_out << "!";
        break;
    case cir::ExprKind::BinaryNegation:
        m_out << "~";
        break;
    case cir::ExprKind::ReductionAnd:
        m_out << "red_and";
        break;
    case cir::ExprKind::ReductionNand:
        m_out << "red_nand";
        break;
    case cir::ExprKind::ReductionOr:
        m_out << "red_or";
        break;
    case cir::ExprKind::ReductionNor:
        m_out << "red_nor";
        break;
    case cir::ExprKind::ReductionXor:
        m_out << "red_xor";
        break;
    case cir::ExprKind::ReductionXnor:
        m_out << "red_xnor";
        break;
    case cir::ExprKind::Posedge:
        m_out << "posedge";
        break;
    case cir::ExprKind::Negedge:
        m_out << "negedge";
        break;
    default:
        m_out << "invalid_unary";
        break;
    }
}

void CirPrinter::printBinaryOp(const cir::ExprKind kind) {
    switch (kind) {
    case cir::ExprKind::Subtraction:
        m_out << "-";
        break;
    case cir::ExprKind::Division:
        m_out << "/";
        break;
    case cir::ExprKind::Modulo:
        m_out << "%";
        break;
    case cir::ExprKind::Equality:
        m_out << "==";
        break;
    case cir::ExprKind::NotEquality:
        m_out << "!=";
        break;
    case cir::ExprKind::GreaterThan:
        m_out << ">";
        break;
    case cir::ExprKind::GreaterThanEq:
        m_out << ">=";
        break;
    case cir::ExprKind::LessThan:
        m_out << "<";
        break;
    case cir::ExprKind::LessThanEq:
        m_out << "<=";
        break;
    case cir::ExprKind::LeftShift:
        m_out << "<<";
        break;
    case cir::ExprKind::RightShift:
        m_out << ">>";
        break;
    case cir::ExprKind::Addition:
        m_out << "+";
        break;
    case cir::ExprKind::Multiplication:
        m_out << "*";
        break;
    case cir::ExprKind::LogicalAnd:
        m_out << "and";
        break;
    case cir::ExprKind::LogicalOr:
        m_out << "or";
        break;
    case cir::ExprKind::BitwiseAnd:
        m_out << "&";
        break;
    case cir::ExprKind::BitwiseOr:
        m_out << "|";
        break;
    case cir::ExprKind::BitwiseXor:
        m_out << "^";
        break;
    case cir::ExprKind::BitwiseXnor:
        m_out << "~^";
        break;
    default:
        m_out << "binary_invalid";
    }
}

void CirPrinter::printIndent() {
    for (size_t i = 0; i < m_indent; i++) {
        m_out << " ";
    }
}

} // namespace cudalator
