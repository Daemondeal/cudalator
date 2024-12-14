#include "SurelogTranslator.hpp"

#include "cir/CIR.h"
#include "uhdm/BaseClass.h"
#include "uhdm/vpi_user.h"
#include <spdlog/spdlog.h>

namespace cudalator {

static cir::Loc getLocFromVpi(const UHDM::BaseClass& obj_h) {
    return cir::Loc(obj_h.VpiLineNo(), obj_h.VpiColumnNo());
}

SurelogTranslator::SurelogTranslator(cir::Ast& ast) : m_ast(ast) {}

cir::SignalIdx SurelogTranslator::parsePort(const UHDM::port& port) {
    auto name = port.VpiName();
    auto loc = getLocFromVpi(port);
    auto direction = port.VpiDirection();

    cir::SignalDirection kind;

    switch (direction) {
    case vpiInput: {
        kind = cir::SignalDirection::Input;
    } break;
    case vpiOutput: {
        kind = cir::SignalDirection::Output;
    } break;
    case vpiInout: {
        kind = cir::SignalDirection::Inout;
    } break;
    default: {
        spdlog::error("Invalid port type for port \"{}\"", name);
    } break;
    };

    auto low_conn = port.Low_conn<UHDM::ref_obj>();

    if (!low_conn) {
        spdlog::warn("Cannot find low_conn for port \"{}\"", name);
        return 0;
    }

    auto logic_net = low_conn->Actual_group<UHDM::logic_net>();

    if (logic_net) {
        // TODO: Implement this
        cir::TypeIdx typ = 0;

        return m_ast.emplaceNode<cir::Signal>(name, loc, 0, kind);
    }

    spdlog::warn("Port {} is not logic. TODO: Add support for more types",
                 name);
    return 0;
}

cir::ModuleIdx SurelogTranslator::parseModule(const UHDM::module_inst& module) {
    auto name = module.VpiName();
    auto loc = getLocFromVpi(module);

    auto mod_idx = m_ast.emplaceNode<cir::Module>(name, loc);
    auto& ast_mod = m_ast.getNode(mod_idx);

    for (auto port : *module.Ports()) {
        auto port_idx = parsePort(*port);
        if (port_idx.isValid()) {
            ast_mod.addSignal(port_idx);
        }
    }

    // TODO: Add the rest

    return mod_idx;
}

} // namespace cudalator
