#pragma once

#include "PassBase.hpp"
#include "cir/CIR.h"
#include <set>

// This pass populates the sensitivity list of each process, visiting all
// statements and expressions to collect all signal, then inserting them inside
// the appropriate sensitivity lists.
namespace cudalator {
class PopulateSensitivityList : public PassBase {
public:
    PopulateSensitivityList(cir::Ast& ast) : PassBase(ast) {}

    void runPass() override;

private:
    void processProcess(cir::Process& proc);
    void processStatement(cir::Statement& statement);
    void processExpr(cir::Expr& expr);

    std::set<cir::SignalIdx> m_collected_signals;

    cir::ScopeIdx m_current_scope;
};
} // namespace cudalator
