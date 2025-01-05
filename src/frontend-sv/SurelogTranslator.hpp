#pragma once

#include "FrontendError.hpp"
#include "cir/CIR.h"

#include <string_view>
#include <uhdm/uhdm.h>

namespace cudalator {
class SurelogTranslator {
public:
    SurelogTranslator(cir::Ast& ast);

    cir::ModuleIdx parseModule(const UHDM::module_inst& module);

    cir::ModulePort parsePort(const UHDM::port& port);

    cir::SignalIdx parseNet(const UHDM::net& net);

    cir::SignalIdx parseVariable(const UHDM::variables& variable);

    cir::ProcessIdx parseAlways(const UHDM::always& proc);

    cir::StatementIdx parseStatement(const UHDM::any *statement);

    cir::StatementIdx parseScope(const UHDM::scope& scope);

    cir::StatementIdx parseAtomicStmt(const UHDM::atomic_stmt& stmt);

    void parseSensitivityList(const UHDM::any* condition, std::vector<cir::SensitivityListElement> &result);

    cir::ProcessIdx parseContinuousAssignment(const UHDM::cont_assign& assign);

    cir::ExprIdx parseExpr(const UHDM::expr& expr);

    cir::TypeIdx parseTypespec(const UHDM::ref_typespec& typespec,
                               std::string_view signal_name);

    std::vector<FrontendError>& getErrors();

private:
    cir::Ast& m_ast;
    std::vector<FrontendError> m_errors;

    cir::ScopeIdx m_current_scope;

    cir::SignalIdx getSignalFromRef(const UHDM::ref_obj& ref);

    void throwError(std::string message, cir::Loc loc);
    void throwErrorTodo(std::string message, cir::Loc loc);
    void throwErrorUnsupported(std::string message, cir::Loc loc);
};
} // namespace cudalator
