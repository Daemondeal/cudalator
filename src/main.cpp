#include "frontend-sv/Entry.hpp"

#include <cir/CIR.h>

#include <functional>
#include <iostream>
#include <vector>

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "USAGE: cudalator-compiler <path-to-sv-file>\n";
        return -1;
    }

    cir::Ast ast;

    auto mod = ast.emplaceNode<cir::Module>("test");
    auto proc = ast.emplaceNode<cir::Process>("proc");
    mod->addProcess(proc);


    for (auto proc : mod->processes) {
        std::cout << proc->name << std::endl;
    }

    std::string source_path(argv[1]);

    // cudalator::compile_sv_to_cil({source_path});

    return 0;
}
