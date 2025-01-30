#pragma once

#include "FrontendError.hpp"
#include "cir/CIR.h"
#include "uhdm/module_inst.h"

#include <string_view>
#include <uhdm/uhdm.h>

namespace cudalator {
class SurelogTranslator {
public:
    SurelogTranslator(cir::Ast& ast);

    void parseModuleFlattened(const UHDM::module_inst& module, cir::ModuleIdx module_idx, bool is_top);

    cir::ModuleIdx parseModule(const UHDM::module_inst& module);

    cir::ModulePort parsePortTop(const UHDM::port& port);

    cir::ProcessIdx parsePortSub(const UHDM::port& port);

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

    std::string m_signals_prefix;

    cir::SignalIdx getSignalFromRef(const UHDM::ref_obj& ref);

    std::string_view cleanSignalName(std::string_view name);

    void throwError(std::string message, cir::Loc loc);
    void throwErrorTodo(std::string message, cir::Loc loc);
    void throwErrorUnsupported(std::string message, cir::Loc loc);
};
} // namespace cudalator
