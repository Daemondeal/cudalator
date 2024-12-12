#include "SurelogParser.hpp"
#include "uhdm/BaseClass.h"
#include "uhdm/design.h"
#include "uhdm/logic_net.h"
#include "uhdm/module_inst.h"
#include "uhdm/port.h"
#include "uhdm/ref_obj.h"
#include "uhdm/sv_vpi_user.h"
#include "uhdm/typespec.h"
#include "uhdm/uhdm_types.h"
#include "uhdm/uhdm_vpi_user.h"
#include "uhdm/vpi_uhdm.h"
#include "uhdm/vpi_user.h"
#include "uhdm/vpi_visitor.h"
#include "utils.hpp"
#include <cstdint>
#include <spdlog/spdlog.h>

namespace cudalator {
SurelogParser::SurelogParser() {}

#define BLACK_MAGIC(value)                                                     \
    reinterpret_cast<UHDM_OBJECT_TYPE>(reinterpret_cast<void *>(value) - 4);

static void parsePort(const UHDM::port *port) {
    spdlog::debug("Parsing Port {}", port->VpiName());

    auto low_conn = port->Low_conn<UHDM::ref_obj>();

    if (!low_conn) {
        spdlog::error("parsePort low conn didn't found a refObject");
        exit(-1);
    }


    auto logic_net = low_conn->Actual_group<UHDM::logic_net>();
    if (logic_net) {
        spdlog::debug("Logic Net Found!");

        auto typespec = logic_net->Typespec();
        if (typespec) {
            auto act = typespec->Actual_typespec<UHDM::logic_typespec>();
            auto rng = act->Ranges();
            if (rng && rng->size() == 1) {
                auto r = rng->at(0);
                auto left = r->Left_expr()->VpiDecompile();
                auto right = r->Right_expr()->VpiDecompile();

                spdlog::debug("Range: [{}:{}]", left, right);
            }
        }

    }
}

static void parseModule(const UHDM::module_inst *module) {
    spdlog::debug("Parsing Module {}", module->VpiName());

    for (auto port : *module->Ports()) {
        parsePort(port);
    }
}

static void parseDesign(const UHDM::design *design) {
    spdlog::debug("Parsing design {}", design->VpiName());
    for (auto mod : *design->TopModules()) {
        parseModule(mod);
    }
}

void SurelogParser::parse(const vpiHandle& handle) {
    auto object_type = vpi_get(vpiType, handle);

    if (object_type == vpiDesign) {
        auto uh = reinterpret_cast<const uhdm_handle *>(handle);
        auto des = reinterpret_cast<const UHDM::design *>(uh->object);

        parseDesign(des);
        return;
    }

    auto uhandle = reinterpret_cast<const uhdm_handle *>(handle);

    auto obj = reinterpret_cast<const UHDM::BaseClass *>(uhandle->object);
    spdlog::debug("Test {}", obj->VpiName());

    if (obj->VpiType() == vpiModule) {
        auto modobj = dynamic_cast<const UHDM::module_inst *>(obj);
        for (auto proc : *modobj->Process()) {
            spdlog::debug("{}", proc->VpiName());
        }
    }

    spdlog::debug("typ {} obj_typ {}", obj->VpiType(), object_type);

    switch (object_type) {
    case vpiDesign: {
        vpiHandle itr = vpi_iterate(UHDM::uhdmtopModules, handle);

        while (vpiHandle child = vpi_scan(itr)) {
            auto child_type = vpi_get(vpiType, child);
            parseModule(child);
        }

        spdlog::debug("Visiting Design Node");
    } break;
    case vpiPort: {
        parsePort(handle);
    } break;
    case vpiModule: {
        spdlog::debug("Visiting Module Node");
    } break;
    default: {
        spdlog::error("Unknown type found: {}", object_type);
    } break;
    }
}

void SurelogParser::parseModule(const vpiHandle& module_h) {
    auto type = vpi_get_str(vpiDefName, module_h);
    auto name = vpi_get_str(vpiName, module_h);
    spdlog::debug("Type: {}, Name: {}", type, name);

    vpiHandle child_h;
    auto iterator = vpi_iterate(vpiPort, module_h);
    while ((child_h = vpi_scan(iterator))) {
        parse(child_h);
    }
}

void SurelogParser::parsePort(const vpiHandle& port_h) {
    auto name = vpi_get_str(vpiName, port_h);
    auto type = vpi_get(vpiPortType, port_h);
    auto size = vpi_get(vpiSize, port_h);

    auto low_conn_h = vpi_handle(vpiLowConn, port_h);
    auto high_conn_h = vpi_handle(vpiHighConn, port_h);

    if (low_conn_h) {
        auto low_conn_name = vpi_get_str(vpiName, low_conn_h);
        spdlog::debug("Low conn present. {}", low_conn_name);

        auto actual_h = vpi_handle(vpiActual, low_conn_h);
        if (!actual_h)
            spdlog::error("Cannot find actual");
        auto type = vpi_get(vpiType, actual_h);

        spdlog::debug("Port type: {}", type);
    }
    if (high_conn_h) {
        auto high_conn_name = vpi_get_str(vpiName, high_conn_h);
        spdlog::debug("Low conn present. {}", high_conn_name);
    }

    spdlog::debug("Name: {}, type: {}, size: {}", name, type, size);

    auto typespec_h = vpi_handle(vpiTypespec, port_h);
    if (typespec_h) {
        auto type_name = vpi_get_str(vpiName, typespec_h);
        spdlog::debug("Typespec Name: {}", type_name);
    }
}

}; // namespace cudalator
