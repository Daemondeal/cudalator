#pragma once

#include <optional>
#include <string>
#include <vector>

#include <fmt/os.h>

#include "../codegen/module.hpp"
#include "Process.hpp"
#include "Vcd.hpp"

using ProcType = Process<StateType>;
using ApplyInputFunc = void (*)(StateType* state, int idx, int cycle);

class Circuit {
public:
    Circuit() = delete;
    Circuit(int number_of_circuits);
    ~Circuit();

    void apply_input(ApplyInputFunc func);
    void eval();

    void open_vcd(const std::string& path, int circuit_idx);
    void dump_to_vcd();

private:
    int m_num_circuits;
    int m_cycles;

    StateType* d_states;
    StateType* d_previous_states;
    DiffType* d_diffs;

    DiffType* h_diffs;

    // cpu only
    std::vector<ProcType> m_processes;
    std::optional<VcdDump> m_vcd;

    void first_eval();
};
