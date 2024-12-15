#pragma once

#include "cir/CIR.h"
#include "uhdm/cont_assign.h"
#include "uhdm/ref_typespec.h"
#include "uhdm/variables.h"
#include <string_view>
#include <uhdm/uhdm.h>

namespace cudalator {
class SurelogTranslator {
public:
    SurelogTranslator(cir::Ast &ast);

    cir::ModuleIdx parseModule(const UHDM::module_inst &module);

    cir::SignalIdx parsePort(const UHDM::port &port);

    cir::SignalIdx parseVariable(const UHDM::variables &variable);

    cir::ProcessIdx parseProcess(const UHDM::process_stmt &proc);

    cir::ProcessIdx parseContinuousAssignment(const UHDM::cont_assign &assign);

    cir::TypeIdx parseTypespec(const UHDM::ref_typespec &typespec, std::string_view signal_name);

private:
    cir::Ast& m_ast;
};
} // namespace cudalator
