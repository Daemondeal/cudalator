#include "CudalatorConfig.hpp"
#include "Exceptions.hpp"
#include "backend/Backend.hpp"
#include "frontend-sv/SystemVerilogFrontend.hpp"

#include <argparse/argparse.hpp>
#include <cir/CIR.h>
#include <exception>
#include <memory>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

bool parseArguments(int argc, const char *argv[], CudalatorConfig *out_cfg) {
    argparse::ArgumentParser program("cudalator-compiler");

    program.add_argument("input-files")
        .help("input source files to compile")
        .nargs(argparse::nargs_pattern::at_least_one);

    program.add_argument("-v", "--verbose")
        .help("enable high verbosity logging")
        .default_value(false)
        .implicit_value(true)
        .nargs(0);

    program.add_argument("--print-uhdm-ast")
        .help("print the UDHM ast before translating it to CIR")
        .default_value(false)
        .implicit_value(true)
        .nargs(0);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        spdlog::error("{}", err.what());
        return false;
    }

    out_cfg->sources = program.get<std::vector<std::string>>("input-files");
    out_cfg->verbose = program.get<bool>("verbose");
    out_cfg->print_udhm_ast = program.get<bool>("--print-uhdm-ast");

    return true;
}

int main(int argc, const char *argv[]) {
    CudalatorConfig cfg;

    // [<levelname>]: <message>
    spdlog::set_pattern("[%^%l%$] %v");

    bool result = parseArguments(argc, argv, &cfg);

    if (!result) {
        return -1;
    }

    if (cfg.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    cudalator::SystemVerilogFrontend frontend;

    try {
        auto ast = frontend.compileSvToCir(cfg.sources, cfg.print_udhm_ast);

        if (!ast) {
            spdlog::error("Error during compilation");
            return -1;
        }

        cudalator::run_backend(std::move(ast), true);
    } catch (cudalator::UnsupportedException& error) {
        auto loc = error.loc();
        spdlog::error("{} Unsupported: {}", loc, error.what());
        return -1;
    } catch (cudalator::UnimplementedException& error) {
        auto loc = error.loc();
        spdlog::error("{} Unimplemented: {}", loc, error.what());
        return -1;
    } catch (cudalator::CompilerException& error) {
        auto loc = error.loc();
        spdlog::error("{} {}", loc, error.what());
        return -1;
    }

    return 0;
}
