#include "SurelogParser.hpp"
#include "uhdm/uhdm_types.h"
#include "uhdm/vpi_user.h"
#include <iostream>
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
        }

        spdlog::debug("Visiting Design Node");
    } break;
    case vpiModule: {
        spdlog::debug("Visiting Module Node");
    } break;
    default: {
        spdlog::error("Unknown type found: {}", object_type);
    } break;
    }
}

}; // namespace cudalator
