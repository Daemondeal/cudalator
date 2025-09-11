#pragma once

#include <optional>
#include <string>

#include <fmt/os.h>

#include "Bit.hpp"
#include "Process.hpp"

#include "Stats.hpp"
#include "../codegen/module.hpp"
#include "Vcd.hpp"

using ProcType = Process<StateType>;

using ApplyInputFunc = void (*)(StateType* state, int idx, int cycle);

class Circuit {
public:
    Circuit() = delete;

    Circuit(int number_of_circuits);

    void apply_input(ApplyInputFunc func);
    void eval();

    StateType &get_state(int idx) {
        return m_states[idx];
    }

    void open_vcd(const std::string path, int circuit_idx);
    void dump_to_vcd();

    const Stats &get_stats() const {
        return m_stats;
    }
private:
    int m_num_circuits;
    int m_cycles;

    Stats m_stats;
    std::vector<ProcType> m_processes;
    std::vector<StateType> m_states;
    std::vector<StateType> m_previous_states;
    std::optional<VcdDump> m_vcd;

    void first_eval();
};

 
