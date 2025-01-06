#include "SurelogTranslator.hpp"

#include "../Exceptions.hpp"
#include "../utils.hpp"

#include "FrontendError.hpp"
#include "cir/CIR.h"
#include "uhdm/int_var.h"
#include "utils.hpp"

#include <charconv>
#include <spdlog/spdlog.h>
#include <string_view>
#include <uhdm/uhdm.h>
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

SurelogTranslator::SurelogTranslator(cir::Ast& ast)
    : m_ast(ast), m_current_scope(cir::ScopeIdx::null()) {}

cir::ModuleIdx SurelogTranslator::parseModule(const UHDM::module_inst& module) {
    auto name = module.VpiName();
    auto loc = getLocFromVpi(module);

    auto mod_scope = m_ast.emplaceNode<cir::Scope>(name, loc, m_current_scope);
    auto mod_idx = m_ast.emplaceNode<cir::Module>(name, loc, mod_scope);

    auto prev_scope = m_current_scope;
    m_current_scope = mod_scope;

    if (module.Nets()) {
        for (auto net : *module.Nets()) {
            auto ast_net = parseNet(*net);

            m_ast.getNode(m_current_scope).addSignal(ast_net);
        }
    }

    if (module.Variables()) {
        for (auto variable : *module.Variables()) {
            auto ast_variable = parseVariable(*variable);
            m_ast.getNode(m_current_scope).addSignal(ast_variable);
        }
    }

    if (module.Ports()) {
        for (auto port : *module.Ports()) {
            auto ast_port = parsePort(*port);
            m_ast.getNode(mod_idx).addPort(ast_port);
        }
    }

    if (module.Cont_assigns()) {
        for (auto assign : *module.Cont_assigns()) {
            auto proc_idx = parseContinuousAssignment(*assign);

            if (proc_idx.isValid()) {
                auto& ast_mod = m_ast.getNode(mod_idx);
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
                auto& ast_mod = m_ast.getNode(mod_idx);
                ast_mod.addProcess(proc_idx);
            }
        }
    }

    // TODO: Add the rest

    m_current_scope = prev_scope;
    return mod_idx;
}

cir::ModulePort SurelogTranslator::parsePort(const UHDM::port& port) {
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
        ast_direction = cir::SignalDirection::Invalid;
    } break;
    };

    auto low_conn = port.Low_conn<UHDM::ref_obj>();
    CD_ASSERT_NONNULL(low_conn);

    auto signal_idx = getSignalFromRef(*low_conn);

    return cir::ModulePort(signal_idx, ast_direction);
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

cir::StatementIdx
SurelogTranslator::parseStatement(const UHDM::any *statement) {
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

    const UHDM::scope *scope_ptr = &scope;

    if (auto begin = dynamic_cast<const UHDM::begin *>(scope_ptr)) {
        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(
            name, loc, cir::StatementKind::Block);

        auto prev_scope = m_current_scope;
        auto scope = m_ast.emplaceNode<cir::Scope>(name, loc, m_current_scope);
        m_current_scope = scope;

        m_ast.getNode(ast_stmt).setScope(scope);

        if (begin->Variables()) {
            for (auto variable : *begin->Variables()) {
                auto var_idx = parseVariable(*variable);
                m_ast.getNode(m_current_scope).addSignal(var_idx);
            }
        }

        if (begin->Stmts()) {
            for (auto statement : *begin->Stmts()) {
                auto parsed = parseStatement(statement);
                auto& stmt = m_ast.getNode(ast_stmt);
                stmt.addStatement(parsed);
            }
        }

        m_current_scope = prev_scope;
        return ast_stmt;
    } else if (auto named_begin =
                   dynamic_cast<const UHDM::named_begin *>(scope_ptr)) {
        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(
            name, loc, cir::StatementKind::Block);

        auto prev_scope = m_current_scope;
        auto scope = m_ast.emplaceNode<cir::Scope>(name, loc, m_current_scope);
        m_current_scope = scope;

        m_ast.getNode(ast_stmt).setScope(scope);

        if (named_begin->Variables()) {
            for (auto variable : *named_begin->Variables()) {
                auto var_idx = parseVariable(*variable);
                m_ast.getNode(m_current_scope).addSignal(var_idx);
            }
        }

        if (named_begin->Stmts()) {
            for (auto statement : *named_begin->Stmts()) {
                auto parsed = parseStatement(statement);
                auto& stmt = m_ast.getNode(ast_stmt);
                stmt.addStatement(parsed);
            }
        }

        m_current_scope = prev_scope;
        return ast_stmt;
    } else if (auto for_stmt =
                   dynamic_cast<const UHDM::for_stmt *>(scope_ptr)) {
        auto merge_multiple_stmts = [&](UHDM::VectorOfany *stmts) {
            CD_ASSERT_NONNULL(stmts);
            CD_ASSERT(stmts->size() > 0);

            auto name = stmts->at(0)->VpiName();
            auto loc = getLocFromVpi(*stmts->at(0));
            auto block_idx = m_ast.emplaceNode<cir::Statement>(
                name, loc, cir::StatementKind::Block);

            for (auto stmt : *stmts) {
                auto ast_stmt = parseStatement(stmt);
                m_ast.getNode(block_idx).addStatement(ast_stmt);
            }

            return block_idx;
        };

        auto prev_scope = m_current_scope;
        auto scope = m_ast.emplaceNode<cir::Scope>(name, loc, m_current_scope);
        m_current_scope = scope;


        if (for_stmt->Variables()) {
            for (auto variable : *for_stmt->Variables()) {
                auto var_idx = parseVariable(*variable);
                m_ast.getNode(m_current_scope).addSignal(var_idx);
            }
        }

        cir::StatementIdx ast_init;
        cir::StatementIdx ast_incr;

        // FIXME: For whatever reason using multiple init statements doesn't really work.
        //        It seems like it's a bug on Surelog's end, but maybe we are doing something 
        //        wrong. Probably worth a look or an issue in their github page.
        auto inits = for_stmt->VpiForInitStmts();
        if (inits) {
            if (inits->size() == 1) {
                ast_init = parseStatement(inits->at(0));
            } else {
                ast_init = merge_multiple_stmts(inits);
            }
        } else {
            CD_ASSERT_NONNULL(for_stmt->VpiForInitStmt());
            ast_init = parseStatement(for_stmt->VpiForInitStmt());
        }

        // NOTE: It's important to parse the condition after the initialization statements,
        //       because sometimes initialization statements contain variable declarations. 
        auto cond = for_stmt->VpiCondition();
        CD_ASSERT_NONNULL(cond);
        auto ast_cond = parseExpr(*cond);

        auto incrs = for_stmt->VpiForIncStmts();
        if (incrs) {
            if (incrs->size() == 1) {
                ast_incr = parseStatement(incrs->at(0));
            } else {
                ast_incr = merge_multiple_stmts(incrs);
            }
        } else {
            CD_ASSERT_NONNULL(for_stmt->VpiForIncStmt());
            ast_incr = parseStatement(for_stmt->VpiForIncStmt());
        }

        auto body = for_stmt->VpiStmt();
        CD_ASSERT_NONNULL(body);
        auto ast_body = parseStatement(body);

        auto kind = cir::StatementKind::For;
        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_init);
        m_ast.getNode(ast_stmt).addStatement(ast_incr);
        m_ast.getNode(ast_stmt).addStatement(ast_body);

        m_ast.getNode(ast_stmt).setScope(m_current_scope);

        m_current_scope = prev_scope;
        return ast_stmt;
    }

    // TODO: Process for and foreach scopes
    throwErrorUnsupported("Unimplemented scope kind", loc);
    auto kind = cir::StatementKind::Invalid;
    return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
}

cir::StatementIdx
SurelogTranslator::parseAtomicStmt(const UHDM::atomic_stmt& stmt) {
    auto name = stmt.VpiName();
    auto loc = getLocFromVpi(stmt);

    auto *stmt_ptr = &stmt;

    // NOTE: At the moment we're not supporting the following statements,
    // described by the LRM:
    //       - (waits)
    //       - delay control
    //       - event control
    //       - event stmt
    //       - assign stmt
    //       - deassign
    //       - (disables)
    //       - (tf call)
    //       - force
    //       - release
    //       - expect stmt
    //       - immediate assert
    //       - immediate assume
    //       - immediate cover

    if (auto assign = dynamic_cast<const UHDM::assignment *>(stmt_ptr)) {
        CD_ASSERT_NONNULL(assign->Lhs());
        CD_ASSERT_NONNULL(assign->Rhs());

        auto *assign_rhs = assign->Rhs<UHDM::expr>();
        CD_ASSERT_MSG(assign_rhs != nullptr,
                      "Interface expr are not supported yet");

        auto lhs = parseExpr(*assign->Lhs());
        auto rhs = parseExpr(*assign_rhs);

        cir::StatementKind kind;
        if (assign->VpiBlocking()) {
            kind = cir::StatementKind::Assignment;
        } else {
            kind = cir::StatementKind::NonBlockingAssignment;
        }

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(lhs);
        m_ast.getNode(ast_stmt).setRhs(rhs);

        return ast_stmt;
    } else if (auto if_stmt = dynamic_cast<const UHDM::if_stmt *>(stmt_ptr)) {
        auto condition = if_stmt->VpiCondition();
        CD_ASSERT_NONNULL(condition);
        auto ast_cond = parseExpr(*condition);

        auto body = if_stmt->VpiStmt();
        CD_ASSERT_NONNULL(body);

        auto ast_body = parseStatement(body);

        auto kind = cir::StatementKind::If;
        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_body);

        return ast_stmt;
    } else if (auto if_else = dynamic_cast<const UHDM::if_else *>(stmt_ptr)) {
        auto condition = if_else->VpiCondition();
        CD_ASSERT_NONNULL(condition);
        auto ast_cond = parseExpr(*condition);

        auto body = if_else->VpiStmt();
        auto else_stmt = if_else->VpiElseStmt();

        CD_ASSERT_NONNULL(body);
        CD_ASSERT_NONNULL(else_stmt);

        auto ast_body = parseStatement(body);
        auto ast_else = parseStatement(else_stmt);

        auto kind = cir::StatementKind::IfElse;

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_body);
        m_ast.getNode(ast_stmt).addStatement(ast_else);

        return ast_stmt;
    } else if (auto while_stmt =
                   dynamic_cast<const UHDM::while_stmt *>(stmt_ptr)) {
        auto condition = while_stmt->VpiCondition();
        CD_ASSERT_NONNULL(condition);

        auto ast_cond = parseExpr(*condition);

        auto body = while_stmt->VpiStmt();
        CD_ASSERT_NONNULL(body);
        auto ast_body = parseStatement(body);
        auto kind = cir::StatementKind::While;

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_body);

        return ast_stmt;
    } else if (auto do_while = dynamic_cast<const UHDM::do_while *>(stmt_ptr)) {
        auto condition = do_while->VpiCondition();
        CD_ASSERT_NONNULL(condition);

        auto ast_cond = parseExpr(*condition);

        auto body = do_while->VpiStmt();
        CD_ASSERT_NONNULL(body);
        auto ast_body = parseStatement(body);
        auto kind = cir::StatementKind::DoWhile;

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_body);

        return ast_stmt;
    } else if (auto repeat = dynamic_cast<const UHDM::repeat *>(stmt_ptr)) {
        auto condition = repeat->VpiCondition();
        CD_ASSERT_NONNULL(condition);
        auto ast_cond = parseExpr(*condition);

        auto body = repeat->VpiStmt();
        CD_ASSERT_NONNULL(body);
        auto ast_body = parseStatement(body);

        auto kind = cir::StatementKind::Repeat;

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_body);

        return ast_stmt;
    } else if (auto case_stmt =
                   dynamic_cast<const UHDM::case_stmt *>(stmt_ptr)) {
        // TODO: Figure this stuff out
        (void)case_stmt;
        throwErrorTodo("Case statements are unimplemented", loc);
        auto kind = cir::StatementKind::Invalid;
        return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
    } else if (auto for_stmt = dynamic_cast<const UHDM::for_stmt *>(stmt_ptr)) {
        auto merge_multiple_stmts = [&](UHDM::VectorOfany *stmts) {
            CD_ASSERT_NONNULL(stmts);
            CD_ASSERT(stmts->size() > 0);

            auto name = stmts->at(0)->VpiName();
            auto loc = getLocFromVpi(*stmts->at(0));
            auto block_idx = m_ast.emplaceNode<cir::Statement>(
                name, loc, cir::StatementKind::Block);

            for (auto stmt : *stmts) {
                auto ast_stmt = parseStatement(stmt);
                m_ast.getNode(block_idx).addStatement(ast_stmt);
            }

            return block_idx;
        };

        auto cond = for_stmt->VpiCondition();
        CD_ASSERT_NONNULL(cond);
        auto ast_cond = parseExpr(*cond);

        cir::StatementIdx ast_init;
        cir::StatementIdx ast_incr;

        auto inits = for_stmt->VpiForInitStmts();
        if (inits) {
            ast_init = merge_multiple_stmts(inits);
        } else {
            CD_ASSERT_NONNULL(for_stmt->VpiForInitStmt());
            ast_init = parseStatement(for_stmt->VpiForInitStmt());
        }

        auto incrs = for_stmt->VpiForIncStmts();
        if (incrs) {
            ast_incr = merge_multiple_stmts(incrs);
        } else {
            CD_ASSERT_NONNULL(for_stmt->VpiForIncStmt());
            ast_incr = parseStatement(for_stmt->VpiForIncStmt());
        }

        auto body = for_stmt->VpiStmt();
        CD_ASSERT_NONNULL(body);
        auto ast_body = parseStatement(body);

        auto kind = cir::StatementKind::For;
        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);
        m_ast.getNode(ast_stmt).addStatement(ast_init);
        m_ast.getNode(ast_stmt).addStatement(ast_incr);
        m_ast.getNode(ast_stmt).addStatement(ast_body);

        spdlog::warn("For found without a scope: line {}, column {}", loc.line,
                     loc.column);

        return ast_stmt;
    } else if (auto foreach =
                   dynamic_cast<const UHDM::foreach_stmt *>(stmt_ptr)) {
        // TODO: Figure this stuff out
        (void)foreach;
        throwErrorTodo("Foreach statements are unimplemented", loc);
        auto kind = cir::StatementKind::Invalid;
        return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
    } else if (auto forever =
                   dynamic_cast<const UHDM::forever_stmt *>(stmt_ptr)) {
        auto body = forever->VpiStmt();
        CD_ASSERT_NONNULL(body);
        auto ast_body = parseStatement(body);

        auto kind = cir::StatementKind::Forever;

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).addStatement(ast_body);

        return ast_stmt;
    } else if (auto return_stmt =
                   dynamic_cast<const UHDM::return_stmt *>(stmt_ptr)) {
        auto condition = return_stmt->VpiCondition();
        CD_ASSERT_NONNULL(condition);
        auto ast_cond = parseExpr(*condition);

        auto kind = cir::StatementKind::Return;

        auto ast_stmt = m_ast.emplaceNode<cir::Statement>(name, loc, kind);

        m_ast.getNode(ast_stmt).setLhs(ast_cond);

        return ast_stmt;
    } else if (auto break_stmt =
                   dynamic_cast<const UHDM::break_stmt *>(stmt_ptr)) {
        (void)break_stmt;
        auto kind = cir::StatementKind::Break;
        return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
    } else if (auto continue_stmt =
                   dynamic_cast<const UHDM::continue_stmt *>(stmt_ptr)) {
        (void)continue_stmt;
        auto kind = cir::StatementKind::Continue;
        return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
    } else if (auto null_stmt =
                   dynamic_cast<const UHDM::null_stmt *>(stmt_ptr)) {
        (void)null_stmt;
        spdlog::warn("Line {} Column {} Null Statement found", loc.line,
                     loc.column);
        auto kind = cir::StatementKind::Null;
        return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
    }

    throwErrorUnsupported("Unimplemented statement kind", loc);
    auto kind = cir::StatementKind::Invalid;
    return m_ast.emplaceNode<cir::Statement>(name, loc, kind);
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
    } else if (dynamic_cast<const UHDM::int_typespec *>(actual)) {
        kind = cir::TypeKind::Int;
    } else {
        std::string owned_name(signal_name);
        throwErrorTodo(string_format("type for signal %s", owned_name.c_str()),
                       loc);
        kind = cir::TypeKind::Invalid;
    }

    auto type_idx = m_ast.emplaceNode<cir::Type>(kind);

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
            auto& type = m_ast.getNode(type_idx);
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
        name, loc, cir::StatementKind::Assignment);

    m_ast.getNode(assignment_idx).setLhs(lhs_idx);
    m_ast.getNode(assignment_idx).setRhs(rhs_idx);

    auto proc_idx = m_ast.emplaceNode<cir::Process>(name, loc, assignment_idx);

    auto& proc = m_ast.getNode(proc_idx);
    proc.setShouldPopulateSensitivityList(true);

    return proc_idx;
}

cir::SignalIdx SurelogTranslator::getSignalFromRef(const UHDM::ref_obj& ref) {
    std::string_view full_name;
    auto loc = getLocFromVpi(ref);
    if (auto net = ref.Actual_group<UHDM::net>()) {
        full_name = net->VpiFullName();
    } else if (auto var = ref.Actual_group<UHDM::variables>()) {
        full_name = var->VpiFullName();
    } else {
        CD_UNREACHABLE("Reference is neither a net nor a variable");
    }

    auto& scope = m_ast.getNode(m_current_scope);
    auto ast_signal = scope.findSignalByName(m_ast, full_name);

    if (!ast_signal.isValid()) {
        std::string owned_name(full_name);
        auto msg = string_format("Undefined signal %s", owned_name.c_str());
        throwError(msg, loc);
        // FIXME: This will probably cause problems, handle it better somehow
        return cir::SignalIdx::null();
    }
    // CD_ASSERT(ast_signal.isValid());
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

    } else if (auto variable = dynamic_cast<const UHDM::variables *>(&expr)) {
        auto full_name = variable->VpiFullName();
        auto ast_signal =
            m_ast.getNode(m_current_scope).findSignalByName(m_ast, full_name);

        // So, as it turns out sometimes Surelog decides to declare signals
        // inside scopes in an expression instead of the vpiVariables field like
        // it always does, like for example inside for statement initializers.
        // This is a bit hacky, but it should be enough to handle that weird
        // behavior.
        if (!ast_signal.isValid()) {
            ast_signal = parseVariable(*variable);
            m_ast.getNode(m_current_scope).addSignal(ast_signal);
        }

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

    auto signal_idx = m_ast.emplaceNode<cir::Signal>(name, loc, full_name, typ,
                                                     cir::SignalLifetime::Net);

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
        name, loc, full_name, typ, cir::SignalLifetime::Static);

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
