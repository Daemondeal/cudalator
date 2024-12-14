#pragma once

#include "cir/CIR.h"
#include <uhdm/uhdm.h>

namespace cudalator {
class SurelogTranslator {
public:
    SurelogTranslator(cir::Ast &ast);

    cir::ModuleIdx parseModule(const UHDM::module_inst &module);

    cir::SignalIdx parsePort(const UHDM::port &port);

private:
    cir::Ast& m_ast;
};
} // namespace cudalator
