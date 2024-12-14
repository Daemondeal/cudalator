#include "backend/CirPrinter.hpp"
#include "frontend-sv/SystemVerilogFrontend.hpp"

#include <cir/CIR.h>
#include <memory>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string_view>
#include <vector>

std::unique_ptr<cir::Ast> generateTestAst() {
    auto ast = std::make_unique<cir::Ast>();
    cir::Loc zero_loc(0, 0);
    std::string_view vw("test");

    auto top_idx = ast->emplaceNode<cir::Module>(vw, zero_loc);

    auto& top = ast->getNode(top_idx);

    auto sig_idx = ast->emplaceNode<cir::Signal>(vw, zero_loc, 0, cir::SignalKind::Input);
    top.addSignal(sig_idx);

    auto proc_idx = ast->emplaceNode<cir::Process>(vw, zero_loc);
    auto& proc = ast->getNode(proc_idx);
    proc.addSignal(sig_idx);
    top.addProcess(proc_idx);

    ast->setTopModule(top_idx);

    return ast;
}

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

    // frontend.compile_sv_to_cil({source_path});

    auto ast = generateTestAst();

    cudalator::CirPrinter printer;
    printer.printAst(*ast);

    spdlog::debug("Test");
    return 0;
}
