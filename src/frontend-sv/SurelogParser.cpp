#include "SurelogParser.hpp"
#include "uhdm/sv_vpi_user.h"
#include "uhdm/uhdm_types.h"
#include "uhdm/vpi_user.h"
#include <spdlog/spdlog.h>

namespace cudalator {
SurelogParser::SurelogParser() {}

void SurelogParser::parse(const vpiHandle& handle) {
    auto object_type = vpi_get(vpiType, handle);

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
