#include "SurelogParser.hpp"
#include "uhdm/uhdm_types.h"
#include "uhdm/vpi_user.h"
#include <iostream>

namespace cudalator {
SurelogParser::SurelogParser() {}

void SurelogParser::parse(const vpiHandle& handle) {
    auto object_type = vpi_get(vpiType, handle);

    switch (object_type) {
    case vpiDesign: {
        vpiHandle itr = vpi_iterate(UHDM::uhdmtopModules, handle);

        while (vpiHandle child = vpi_scan(itr)) {
            auto child_type = vpi_get(vpiType, child);
            std::cout << child_type << std::endl;
        }

        std::cout << "Design\n";
    } break;
    case vpiModule: {
        std::cout << "Module\n";
    } break;
    default: {
        std::cout << "Unknown type " << object_type << "\n";
    } break;
    }

    std::cout << "This is a test\n";
}

}; // namespace cudalator
