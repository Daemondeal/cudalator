#include "frontend-sv/SystemVerilogFrontend.hpp"

#include <cir/CIR.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <vector>

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "USAGE: cudalator-compiler <path-to-sv-file>\n";
        return -1;
    }

    // TODO: Make this configurable
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%^%l%$] %v");

    std::string source_path(argv[1]);

    cudalator::SystemVerilogFrontend frontend;

    frontend.compile_sv_to_cil({source_path});

    return 0;
}
