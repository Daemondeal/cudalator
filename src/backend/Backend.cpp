#include "Backend.hpp"
#include "CirPrinter.hpp"
#include "PassManager.hpp"

#include <fmt/core.h>

namespace cudalator {
void run_backend(std::unique_ptr<cir::Ast> ast, bool print_ast) {
    PassManager manager;
    manager.runPasses(*ast);

    if (print_ast) {
        cudalator::CirPrinter printer;
        printer.printAst(*ast);
    }
}
} // namespace cudalator
