#include "Backend.hpp"
#include "CirPrinter.hpp"
#include "CodeGenerator.hpp"
#include "PassManager.hpp"

#include <fmt/core.h>

#include <filesystem>

namespace cudalator {

namespace fs = std::filesystem;

void run_backend(std::unique_ptr<cir::Ast> ast, bool print_ast) {
    PassManager manager;
    manager.runPasses(*ast);

    if (print_ast) {
        cudalator::CirPrinter printer;
        printer.printAst(*ast);
    }

    // Create output dir if it doesn't exist
    fs::create_directories("build/output");

    auto sourcePath = "build/output/module.cpp";
    auto headerPath = "build/output/module.hpp";

    CodeGenerator::generateCode(*ast, headerPath, sourcePath);

}
} // namespace cudalator
