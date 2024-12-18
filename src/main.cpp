#include "Exceptions.hpp"
#include "backend/CirPrinter.hpp"
#include "frontend-sv/SystemVerilogFrontend.hpp"

#include <cir/CIR.h>
#include <memory>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

std::unique_ptr<cir::Ast> generateTestAst() {
    auto ast = std::make_unique<cir::Ast>();
    cir::Loc zero_loc(0, 0);
    std::string_view vw("test");

    auto top_idx = ast->emplaceNode<cir::Module>(vw, zero_loc);

    auto& top = ast->getNode(top_idx);

    auto sig_idx = ast->emplaceNode<cir::Signal>(
        vw, zero_loc, cir::TypeIdx::null(), cir::SignalDirection::Input);
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

    try {
        auto ast = frontend.compileSvToCir({source_path}, true);

        if (!ast) {
            spdlog::error("Error while compiling \"{}\".", source_path);
            return -1;
        }

        cudalator::CirPrinter printer;
        printer.printAst(*ast);

    } catch (cudalator::UnsupportedException error) {
        auto loc = error.loc();
        spdlog::error("[{}:{}] Unsupported: {}", loc.line, loc.column, error.what());
        return -1;
    } catch (cudalator::UnimplementedException error) {
        auto loc = error.loc();
        spdlog::error("[{}:{}] Unimplemented: {}", loc.line, loc.column, error.what());
        return -1;
    } catch (cudalator::CompilerException error) {
        auto loc = error.loc();
        spdlog::error("[{}:{}] {}", loc.line, loc.column, error.what());
        return -1;
    }

    // auto ast = generateTestAst();

    return 0;
}
