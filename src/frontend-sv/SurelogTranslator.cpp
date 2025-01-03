#include "SurelogTranslator.hpp"

#include "../Exceptions.hpp"
#include "../utils.hpp"

#include "FrontendError.hpp"
#include "cir/CIR.h"
#include "utils.hpp"

#include <uhdm/uhdm.h>
#include <charconv>
#include <spdlog/spdlog.h>
#include <string_view>
#include <vector>

namespace cudalator {

static cir::Loc getLocFromVpi(const UHDM::BaseClass& obj_h) {
    return cir::Loc(obj_h.VpiLineNo(), obj_h.VpiColumnNo());
}

// FIXME: This is not the proper way to do this, implement this properly.
static int64_t evaluateConstant(std::string_view constant, cir::Loc loc) {
    auto pos = constant.find(':');
    if (pos == std::string_view::npos) {
        throw CompilerException("Invalid constant type", loc);
    }

    std::string_view prefix = constant.substr(0, pos);
    std::string_view value_str = constant.substr(pos + 1);
    int64_t result = 0;

    if (prefix == "UINT") {
        uint64_t temp;
        auto [ptr, ec] =
            std::from_chars(value_str.begin(), value_str.end(), temp, 10);
        if (ec != std::errc()) {
            throw CompilerException("Invalid UINT constant value", loc);
        }
        result = static_cast<int64_t>(temp);
    } else if (prefix == "INT") {
        auto [ptr, ec] =
            std::from_chars(value_str.begin(), value_str.end(), result, 10);
        if (ec != std::errc()) {
            throw CompilerException("Invalid INT constant value", loc);
        }
    } else if (prefix == "HEX") {
        auto [ptr, ec] =
            std::from_chars(value_str.begin(), value_str.end(), result, 16);
        if (ec != std::errc()) {
            throw CompilerException("Invalid HEX constant value", loc);
        }
    } else {
        throw CompilerException("Unknown constant type", loc);
    }

    return result;
}

SurelogTranslator::SurelogTranslator(cir::Ast& ast) : m_ast(ast) {}

cir::ModuleIdx SurelogTranslator::parseModule(const UHDM::module_inst& module) {
    auto name = module.VpiName();
    auto loc = getLocFromVpi(module);

    auto mod_idx = m_ast.emplaceNode<cir::Module>(name, loc);
    auto& ast_mod = m_ast.getNode(mod_idx);

    if (module.Nets()) {
        for (auto net : *module.Nets()) {
            auto ast_net = parseNet(*net);
            ast_mod.addSignal(ast_net);
        }
    }

    if (module.Variables()) {
        for (auto variable : *module.Variables()) {
            auto ast_variable = parseVariable(*variable);
            ast_mod.addSignal(ast_variable);
        }
    }

    if (module.Ports()) {
        for (auto port : *module.Ports()) {
            parsePort(*port);
        }
    }

    if (module.Cont_assigns()) {
        for (auto assign : *module.Cont_assigns()) {
            auto proc_idx = parseContinuousAssignment(*assign);

            if (proc_idx.isValid()) {
                ast_mod.addProcess(proc_idx);
            }
        }
    }

    if (module.Process()) {
        for (auto process : *module.Process()) {
            cir::ProcessIdx proc_idx(cir::ProcessIdx::null());

            if (auto always = dynamic_cast<const UHDM::always *>(process)) {
                proc_idx = parseAlways(*always);
            } else {
                spdlog::warn("Only always processes are implemented, skipping "
                             "non-always process.");
            }

            if (proc_idx.isValid()) {
                ast_mod.addProcess(proc_idx);
            }
        }
    }

    // TODO: Add the rest

    return mod_idx;
}

void SurelogTranslator::parsePort(const UHDM::port& port) {
    auto name = port.VpiName();
    auto loc = getLocFromVpi(port);
    auto direction = port.VpiDirection();

    cir::SignalDirection ast_direction;

    switch (direction) {
    case vpiInput: {
        ast_direction = cir::SignalDirection::Input;
    } break;
    case vpiOutput: {
        ast_direction = cir::SignalDirection::Output;
    } break;
    case vpiInout: {
        ast_direction = cir::SignalDirection::Inout;
    } break;
    default: {
        throwError(string_format("Invalid port type for port %s", name), loc);
        return;
    } break;
    };

    auto low_conn = port.Low_conn<UHDM::ref_obj>();
    CD_ASSERT_NONNULL(low_conn);

    auto signal_idx = getSignalFromRef(*low_conn);
    auto& signal = m_ast.getNode(signal_idx);
    signal.setDirection(ast_direction);
}

cir::ProcessIdx SurelogTranslator::parseAlways(const UHDM::always& proc) {
    auto name = proc.VpiName();
    auto loc = getLocFromVpi(proc);

    auto type = proc.VpiAlwaysType();

    // All processes should have at least one statement
    CD_ASSERT_NONNULL(proc.Stmt());

    const UHDM::any *stmt = proc.Stmt();

    std::vector<cir::SensitivityListElement> sensitivity;
    if (auto event_control = proc.Stmt<UHDM::event_control>()) {
        parseSensitivityList(event_control->VpiCondition(), sensitivity);

        stmt = event_control->Stmt();
    }

    auto ast_stmt = parseStatement(stmt);

    auto proc_idx = m_ast.emplaceNode<cir::Process>(name, loc, ast_stmt);
    auto& ast_proc = m_ast.getNode(proc_idx);
    ast_proc.setSensitivityList(std::move(sensitivity));

    if (type == vpiAlwaysComb) {
        ast_proc.setShouldPopulateSensitivityList(true);
    }

    return proc_idx;
}

void SurelogTranslator::parseSensitivityList(
    const UHDM::any *condition,
    std::vector<cir::SensitivityListElement>& result) {

    auto op = dynamic_cast<const UHDM::operation *>(condition);
    auto ref = dynamic_cast<const UHDM::ref_obj *>(condition);

    if (op) {
        auto op_type = op->VpiOpType();

        if (op_type == vpiListOp) {
            for (auto operand : *op->Operands()) {
                parseSensitivityList(operand, result);
            }
        } else if (op_type == vpiNegedgeOp || op_type == vpiPosedgeOp) {
            auto kind = op_type == vpiNegedgeOp ? cir::SensitivityKind::Negedge
                                                : cir::SensitivityKind::Posedge;

            auto operands = op->Operands();
            CD_ASSERT_NONNULL(operands);

            auto first_operand = operands->at(0);
            CD_ASSERT_NONNULL(first_operand);

            auto ref = dynamic_cast<const UHDM::ref_obj *>(first_operand);
            CD_ASSERT_NONNULL(ref);

            auto signal = getSignalFromRef(*ref);
            auto elem = cir::SensitivityListElement(signal, kind);
            result.push_back(elem);
        }
    } else if (ref) {
        auto signal = getSignalFromRef(*ref);
        auto elem =
            cir::SensitivityListElement(signal, cir::SensitivityKind::OnChange);
        result.push_back(elem);
    } else {
        CD_UNREACHABLE(
            "Sensitivity list element was neither an op nor a ref_obj");
    }
}

cir::StatementIdx SurelogTranslator::parseStatement(const UHDM::any *statement) {
    if (auto scope = dynamic_cast<const UHDM::scope *>(statement)) {
        return parseScope(*scope);
    } else if (auto atomic_stmt =
                   dynamic_cast<const UHDM::atomic_stmt *>(statement)) {
        return parseAtomicStmt(*atomic_stmt);
    } else {
        CD_UNREACHABLE("Statement is neither a scope nor an atomic stmt");
        return cir::StatementIdx::null();
    }
}

cir::StatementIdx SurelogTranslator::parseScope(const UHDM::scope& scope) {
    auto name = scope.VpiName();
    auto loc = getLocFromVpi(scope);

    cir::StatementIdx ast_stmt;

    const UHDM::scope *scope_ptr = &scope;

    if (auto begin = dynamic_cast<const UHDM::begin *>(scope_ptr)) {
        ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, cir::StatementKind::Block);
        auto &stmt = m_ast.getNode(ast_stmt);

        if (begin->Stmts()) {
            for (auto statement : *begin->Stmts()) {
                spdlog::info("Adding statement");
                stmt.addStatement(parseStatement(statement));
            }
        }
        spdlog::info("block has {} stmts", stmt.statements().size());
    } else if (auto named_begin =
                   dynamic_cast<const UHDM::named_begin *>(scope_ptr)) {
        ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, cir::StatementKind::Block);
        auto &stmt = m_ast.getNode(ast_stmt);

        if (begin->Stmts()) {
            for (auto statement : *begin->Stmts()) {
                stmt.addStatement(parseStatement(statement));
            }
        }
    }


    return ast_stmt;
}

cir::StatementIdx
SurelogTranslator::parseAtomicStmt(const UHDM::atomic_stmt& scope) {
    auto name = scope.VpiName();
    auto loc = getLocFromVpi(scope);

    auto kind = cir::StatementKind::Invalid;

    auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

    spdlog::warn("todo: parseAtomicStmt");
    return ast_stmt;
}

cir::TypeIdx
SurelogTranslator::parseTypespec(const UHDM::ref_typespec& typespec,
                                 std::string_view signal_name) {
    cir::TypeKind kind;

    auto actual = typespec.Actual_typespec();
    auto name = actual->VpiName();
    auto loc = getLocFromVpi(*actual);

    if (dynamic_cast<const UHDM::logic_typespec *>(actual)) {
        kind = cir::TypeKind::Logic;
    } else if (dynamic_cast<const UHDM::bit_typespec *>(actual)) {
        kind = cir::TypeKind::Bit;
    } else if (dynamic_cast<const UHDM::integer_typespec *>(actual)) {
        kind = cir::TypeKind::Integer;
    } else {
        throwErrorTodo(string_format("type for signal %s", signal_name), loc);
        kind = cir::TypeKind::Invalid;
    }

    auto type_idx = m_ast.emplaceNode<cir::Type>(kind);
    auto& type = m_ast.getNode(type_idx);

    // Ranges

    auto addRangeIfExists = [&](const UHDM::VectorOfrange *ranges) {
        if (!ranges) {
            return;
        }
        for (auto range : *ranges) {
            auto lhs = range->Left_expr<UHDM::constant>();
            if (!lhs) {
                throwErrorTodo("Non constant ranges are not supported yet",
                               loc);
                continue;
            }

            auto rhs = range->Right_expr<UHDM::constant>();
            if (!rhs) {
                throwErrorTodo("Non constant ranges are not supported yet",
                               loc);
                continue;
            }

            auto lhs_val = evaluateConstant(lhs->VpiValue(), loc);
            auto rhs_val = evaluateConstant(rhs->VpiValue(), loc);

            auto ast_range = cir::Range(lhs_val, rhs_val);

            spdlog::debug("Adding range [{}:{}] to {}", lhs_val, rhs_val,
                          signal_name);
            type.addRange(ast_range);
        }
    };

    if (auto logic = dynamic_cast<const UHDM::logic_typespec *>(actual)) {
        addRangeIfExists(logic->Ranges());
    } else if (auto bit = dynamic_cast<const UHDM::bit_typespec *>(actual)) {
        addRangeIfExists(bit->Ranges());
    } else {
        spdlog::debug("Type {} does not need ranges (or ranges are not "
                      "implemented for its type yet)",
                      name);
    }

    return type_idx;
}

cir::ProcessIdx
SurelogTranslator::parseContinuousAssignment(const UHDM::cont_assign& assign) {
    CD_ASSERT_NONNULL(assign.Lhs());
    CD_ASSERT_NONNULL(assign.Rhs());

    auto lhs_idx = parseExpr(*assign.Lhs());
    auto rhs_idx = parseExpr(*assign.Rhs());

    auto name = assign.VpiName();
    auto loc = getLocFromVpi(assign);

    auto assignment_idx = m_ast.emplaceNode<cir::Statement>(
        name, loc, cir::StatementKind::Assignment, lhs_idx, rhs_idx);

    auto proc_idx = m_ast.emplaceNode<cir::Process>(name, loc, assignment_idx);

    auto& proc = m_ast.getNode(proc_idx);
    proc.setShouldPopulateSensitivityList(true);

    return proc_idx;
}

cir::SignalIdx SurelogTranslator::getSignalFromRef(const UHDM::ref_obj& ref) {
    std::string_view full_name;
    if (auto net = ref.Actual_group<UHDM::net>()) {
        full_name = net->VpiFullName();
    } else if (auto var = ref.Actual_group<UHDM::variables>()) {
        full_name = var->VpiFullName();
    } else {
        CD_UNREACHABLE("Reference is neither a net nor a variable");
    }

    auto ast_signal = m_ast.findSignal(full_name);
    CD_ASSERT(ast_signal.isValid());
    return ast_signal;
}

cir::ExprIdx SurelogTranslator::parseExpr(const UHDM::expr& expr) {
    auto name = expr.VpiName();
    auto loc = getLocFromVpi(expr);

    if (auto constant = dynamic_cast<const UHDM::constant *>(&expr)) {
        auto val = expr.VpiValue();
        // TODO: This should be able to handle bigger constants
        auto int_val = evaluateConstant(val, loc);
        auto size = static_cast<uint32_t>(constant->VpiSize());

        auto ast_const =
            m_ast.emplaceNode<cir::Constant>(val, loc, size, int_val);
        if (size > 64) {
            throwErrorTodo(
                "cannot handle constants bigger than 64 bits (for now)", loc);
            return m_ast.emplaceNode<cir::Expr>(name, loc,
                                                cir::ExprKind::Constant);
        }

        return m_ast.emplaceNode<cir::Expr>(name, loc, cir::ExprKind::Constant,
                                            ast_const);
    } else if (auto op = dynamic_cast<const UHDM::operation *>(&expr)) {
        auto operands = op->Operands();

        auto unary_op = vpiUnaryOp(op->VpiOpType());
        if (unary_op != cir::ExprKind::Invalid) {
            CD_ASSERT_NONNULL(operands);
            auto operand = dynamic_cast<UHDM::expr *>(operands->at(0));
            CD_ASSERT_NONNULL(operand);

            auto ast_operand = parseExpr(*operand);
            return m_ast.emplaceNode<cir::Expr>(name, loc, unary_op,
                                                ast_operand);
        }

        auto binary_op = vpiBinaryOp(op->VpiOpType());
        if (binary_op != cir::ExprKind::Invalid) {
            CD_ASSERT_NONNULL(operands);

            auto lhs = dynamic_cast<UHDM::expr *>(operands->at(0));
            auto rhs = dynamic_cast<UHDM::expr *>(operands->at(1));

            CD_ASSERT_NONNULL(lhs);
            CD_ASSERT_NONNULL(rhs);

            auto ast_lhs = parseExpr(*lhs);
            auto ast_rhs = parseExpr(*rhs);

            return m_ast.emplaceNode<cir::Expr>(name, loc, binary_op, ast_lhs,
                                                ast_rhs);
        }

        switch (op->VpiOpType()) {
        default: {
            throwErrorTodo(string_format("operation type %d", op->VpiOpType()),
                           loc);
            return m_ast.emplaceNode<cir::Expr>(name, loc,
                                                cir::ExprKind::Invalid);
        } break;
        }

    } else if (auto part_sel = dynamic_cast<const UHDM::part_select *>(&expr)) {
        // NOTE: Part selects are a subclass of ref_objs, so check for these
        // before checking those
        CD_ASSERT_NONNULL(part_sel->Left_range());
        CD_ASSERT_NONNULL(part_sel->Right_range());

        auto lhs = parseExpr(*part_sel->Left_range());
        auto rhs = parseExpr(*part_sel->Right_range());

        auto ast_signal = getSignalFromRef(*part_sel);
        return m_ast.emplaceNode<cir::Expr>(
            name, loc, cir::ExprKind::PartSelect, lhs, rhs, ast_signal);
    } else if (auto bit_sel = dynamic_cast<const UHDM::bit_select *>(&expr)) {
        CD_ASSERT_NONNULL(bit_sel->VpiIndex());

        auto index = parseExpr(*bit_sel->VpiIndex());
        auto ast_signal = getSignalFromRef(*bit_sel);

        auto expr_idx = m_ast.emplaceNode<cir::Expr>(
            name, loc, cir::ExprKind::BitSelect, ast_signal);

        auto& ast_expr = m_ast.getNode(expr_idx);
        ast_expr.addExpr(index);

        return expr_idx;
    } else if (auto ref_obj = dynamic_cast<const UHDM::ref_obj *>(&expr)) {
        auto ast_signal = getSignalFromRef(*ref_obj);

        return m_ast.emplaceNode<cir::Expr>(name, loc, cir::ExprKind::SignalRef,
                                            ast_signal);

    } else {
        throwErrorTodo("expression type", loc);
        return m_ast.emplaceNode<cir::Expr>(name, loc, cir::ExprKind::Invalid);
    }

    CD_UNREACHABLE("all instruction types should be handled");
    return {};
}

cir::SignalIdx SurelogTranslator::parseNet(const UHDM::net& net) {
    auto name = net.VpiName();
    auto loc = getLocFromVpi(net);
    auto full_name = net.VpiFullName();

    // Hopefully this always exists
    auto type_ref = net.Typespec();
    CD_ASSERT_MSG(!!type_ref, "Found net without typespec.");

    if (net.Ports()) {
        spdlog::warn("Net {} has ports", name);
    }

    auto typ = parseTypespec(*type_ref, name);

    auto signal_idx = m_ast.emplaceNode<cir::Signal>(
        name, loc, full_name, typ, cir::SignalDirection::Internal);

    return signal_idx;
}

cir::SignalIdx
SurelogTranslator::parseVariable(const UHDM::variables& variable) {
    auto name = variable.VpiName();
    auto loc = getLocFromVpi(variable);
    auto full_name = variable.VpiFullName();

    // Hopefully this always exists
    auto type_ref = variable.Typespec();
    CD_ASSERT_MSG(!!type_ref, "Found variable without typespec.");

    auto typ = parseTypespec(*type_ref, name);

    auto signal_idx = m_ast.emplaceNode<cir::Signal>(
        name, loc, full_name, typ, cir::SignalDirection::Internal);

    return signal_idx;
}

std::vector<FrontendError>& SurelogTranslator::getErrors() {
    return m_errors;
}

void SurelogTranslator::throwError(std::string message, cir::Loc loc) {
    m_errors.push_back(FrontendError::other(message, loc));
}

void SurelogTranslator::throwErrorTodo(std::string message, cir::Loc loc) {
    m_errors.push_back(FrontendError::todo(message, loc));
}

void SurelogTranslator::throwErrorUnsupported(std::string message,
                                              cir::Loc loc) {
    m_errors.push_back(FrontendError::unsupported(message, loc));
}

} // namespace cudalator
