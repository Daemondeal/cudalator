#include "Exceptions.hpp"
#include "backend/Backend.hpp"
#include "frontend-sv/SystemVerilogFrontend.hpp"

#include <cir/CIR.h>
#include <memory>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>
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

    try {
        auto ast = frontend.compileSvToCir({source_path}, true);

        if (!ast) {
            spdlog::error("Error while compiling \"{}\".", source_path);
            return -1;
        }

        cudalator::run_backend(std::move(ast), true);
    } catch (cudalator::UnsupportedException error) {
        auto loc = error.loc();
        spdlog::error("(line {}: col {}) Unsupported: {}", loc.line, loc.column,
                      error.what());
        return -1;
    } catch (cudalator::UnimplementedException error) {
        auto loc = error.loc();
        spdlog::error("[line {}: col {}] Unimplemented: {}", loc.line, loc.column,
                      error.what());
        return -1;
    } catch (cudalator::CompilerException error) {
        auto loc = error.loc();
        spdlog::error("(line {}:col {}) {}", loc.line, loc.column, error.what());
        return -1;
    }

    // auto ast = generateTestAst();

    return 0;
}
