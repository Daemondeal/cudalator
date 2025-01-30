#include "PopulateSensitivityList.hpp"
#include "cir/CIR.h"
#include <spdlog/spdlog.h>

namespace cudalator {
void PopulateSensitivityList::runPass() {
    spdlog::debug("Running pass PopulateSensitivityList");


    auto& top = m_ast.getTopModule();
    // NOTE: We only care about variables in the module's scope, so we never 
    //       actually need to update m_current_scope with any block's scope.
    m_current_scope = top.scope();

    for (auto process_idx : top.processes()) {
        auto& process = m_ast.getNode(process_idx);
        if (process.shouldPopulateSensitivityList()) {
            processProcess(process);
        }
    }
}

void PopulateSensitivityList::processProcess(cir::Process& proc) {
    m_collected_signals.clear();

    if (proc.statement().isValid()) {
        auto& ast_statement = m_ast.getNode(proc.statement());
        processStatement(ast_statement);
    }

    // FIXME: This does not handle the case in which the process already had a
    //        sensitivity list, in that case signals could be duplicated.
    //        We should check if that is even possible with always_comb or if it
    //        should be a compiler error.
    for (auto idx : m_collected_signals) {
        proc.addToSensitivityList(idx, cir::SensitivityKind::OnChange);
    }
}

void PopulateSensitivityList::processStatement(cir::Statement& statement) {
    for (auto& sub_stmt_idx : statement.statements()) {
        auto& sub_stmt = m_ast.getNode(sub_stmt_idx);
        processStatement(sub_stmt);
    }

    // TODO: Check if there are more statement kinds that have a "left side"
    //       that should not be inside a sensitivity list
    if (statement.lhs().isValid() &&
        statement.kind() != cir::StatementKind::Assignment &&
        statement.kind() != cir::StatementKind::NonBlockingAssignment) {
        auto& lhs = m_ast.getNode(statement.lhs());
        processExpr(lhs);
    }

    if (statement.rhs().isValid()) {
        auto& rhs = m_ast.getNode(statement.rhs());
        processExpr(rhs);
    }
}

void PopulateSensitivityList::processExpr(cir::Expr& expr) {
    // NOTE: There might be signals found that cannot be found, those are signals
    //       outside the block's scope so we don't care.
    if (expr.signal().isValid()) {
        m_collected_signals.insert(expr.signal());
    }

    for (auto sub_expr_idx : expr.exprs()) {
        auto& sub_expr = m_ast.getNode(sub_expr_idx);
        processExpr(sub_expr);
    }
}

} // namespace cudalator
