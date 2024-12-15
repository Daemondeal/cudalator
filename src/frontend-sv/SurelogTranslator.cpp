#include "SurelogTranslator.hpp"

#include "../utils.hpp"

#include "cir/CIR.h"
#include "uhdm/BaseClass.h"
#include "uhdm/bit_typespec.h"
#include "uhdm/cont_assign.h"
#include "uhdm/integer_typespec.h"
#include "uhdm/logic_typespec.h"
#include "uhdm/vpi_user.h"

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
    CD_ASSERT(!!low_conn, "Port has no associated low_conn .");

    if (!low_conn) {
        spdlog::warn("Cannot find low_conn for port \"{}\"", name);
        return cir::SignalIdx::null();
    }

    auto net = low_conn->Actual_group<UHDM::net>();
    CD_ASSERT(!!net, "Port has no associated low_conn net.");

    auto typespec = net->Typespec();
    CD_ASSERT(!!net, "Port has no associated low_conn net typespec.");
    auto typ = parseTypespec(*typespec, name);

    return m_ast.emplaceNode<cir::Signal>(name, loc, typ, kind);
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

    if (module.Variables()) {
        for (auto variable : *module.Variables()) {
            auto ast_variable = parseVariable(*variable);
            ast_mod.addSignal(ast_variable);
        }
    }

    if (module.Nets()) {
        for (auto net : *module.Nets()) {
            auto name = net->VpiName();

            spdlog::debug("Found net {}", name);
        }
    }

    if (module.Cont_assigns()) {
        for (auto assign : *module.Cont_assigns()) {
            auto proc_idx = parseContinuousAssignment(*assign);

            if (proc_idx.isValid()) {
                spdlog::debug("Valid Proc");
                ast_mod.addProcess(proc_idx);
            }
        }
    }

    // TODO: Add the rest

    return mod_idx;
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
        spdlog::error("Unimplemented type for signal {}", signal_name);
        exit(-1);
    }

    auto type_idx = m_ast.addNode<cir::Type>(kind);

    spdlog::warn("TODO: IMPLEMENT RANGES");

    return type_idx;
}

cir::ProcessIdx
SurelogTranslator::parseContinuousAssignment(const UHDM::cont_assign& assign) {

    return {};
}

cir::SignalIdx
SurelogTranslator::parseVariable(const UHDM::variables& variable) {
    auto name = variable.VpiName();
    auto loc = getLocFromVpi(variable);

    // Hopefully this always exists
    auto type_ref = variable.Typespec();
    CD_ASSERT(!!type_ref, "Found variable without typespec.");

    auto typ = parseTypespec(*type_ref, name);

    auto signal_idx = m_ast.emplaceNode<cir::Signal>(
        name, loc, typ, cir::SignalDirection::Internal);

    return signal_idx;
}

} // namespace cudalator
