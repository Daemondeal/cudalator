#include "SurelogTranslator.hpp"

#include "../Exceptions.hpp"
#include "../utils.hpp"

#include "cir/CIR.h"
#include "uhdm/BaseClass.h"
#include "uhdm/bit_typespec.h"
#include "uhdm/constant.h"
#include "uhdm/cont_assign.h"
#include "uhdm/containers.h"
#include "uhdm/integer_typespec.h"
#include "uhdm/logic_typespec.h"
#include "uhdm/vpi_user.h"
#include <charconv>
#include <spdlog/spdlog.h>
#include <string_view>

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
    CD_ASSERT_MSG(!!low_conn, "Port has no associated low_conn .");

    if (!low_conn) {
        spdlog::warn("Cannot find low_conn for port \"{}\"", name);
        return cir::SignalIdx::null();
    }

    auto net = low_conn->Actual_group<UHDM::net>();
    CD_ASSERT_MSG(!!net, "Port has no associated low_conn net.");

    auto typespec = net->Typespec();
    CD_ASSERT_MSG(!!net, "Port has no associated low_conn net typespec.");
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

            // This is a workaround for the fact that net->Ports() seems to
            // never return a valid pointer, even when the signal is a port.
            // FIXME: Try to debug this better, maybe there's something we don't
            // understand
            if (!m_ast.existsWithName<cir::Signal>(name)) {
                auto ast_net = parseNet(*net);
                ast_mod.addSignal(ast_net);
            } else {
                spdlog::debug("Skipping {} as it already exists", name);
            }
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
        throw UnimplementedException(
            string_format("type for signal %s", signal_name), loc);
    }

    auto type_idx = m_ast.addNode<cir::Type>(kind);
    auto type = m_ast.getNode(type_idx);

    // Ranges

    auto addRangeIfExists = [&type, &loc](const UHDM::VectorOfrange *ranges) {
        if (!ranges) {
            return;
            for (auto range : *ranges) {
                auto lhs = range->Left_expr<UHDM::constant>();
                if (!lhs) {
                    throw UnimplementedException(
                        "Non constant ranges are not supported yet", loc);
                }

                auto rhs = range->Right_expr<UHDM::constant>();
                if (!rhs) {
                    throw UnimplementedException(
                        "Non constant ranges are not supported yet", loc);
                }

                auto lhs_val = evaluateConstant(lhs->VpiValue(), loc);
                auto rhs_val = evaluateConstant(rhs->VpiValue(), loc);

                auto ast_range = cir::Range(lhs_val, rhs_val);

                type.addRange(ast_range);
            }
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

    return {};
}

cir::SignalIdx SurelogTranslator::parseNet(const UHDM::net& net) {
    auto name = net.VpiName();
    auto loc = getLocFromVpi(net);

    // Hopefully this always exists
    auto type_ref = net.Typespec();
    CD_ASSERT_MSG(!!type_ref, "Found net without typespec.");

    if (net.Ports()) {
        spdlog::warn("Net {} has ports", name);
    }

    auto typ = parseTypespec(*type_ref, name);

    auto signal_idx = m_ast.emplaceNode<cir::Signal>(
        name, loc, typ, cir::SignalDirection::Internal);

    return signal_idx;
}

cir::SignalIdx
SurelogTranslator::parseVariable(const UHDM::variables& variable) {
    auto name = variable.VpiName();
    auto loc = getLocFromVpi(variable);

    // Hopefully this always exists
    auto type_ref = variable.Typespec();
    CD_ASSERT_MSG(!!type_ref, "Found variable without typespec.");

    auto typ = parseTypespec(*type_ref, name);

    auto signal_idx = m_ast.emplaceNode<cir::Signal>(
        name, loc, typ, cir::SignalDirection::Internal);

    return signal_idx;
}

} // namespace cudalator
