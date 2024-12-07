#pragma once

#include "GenericAst.h"
#include <string>
#include <string_view>
namespace cir {

struct Signal {
    std::string_view name;
};

struct Process {
    std::string_view name;

    Process(std::string_view name) : name(name) {}
};

struct Module {
public:
    std::string_view name;

    std::vector<Process *> processes;

    Module(std::string_view name) : name(name), processes() {}

    void addProcess(Process *proc) {
        processes.push_back(proc);
    }
};

struct Ast : GenericAst<Signal, Process, Module> {
    using GenericAst::GenericAst;
};

} // namespace cir
