#pragma once

#include "Bit.hpp"
#include "Process.hpp"

#include "../codegen/module.hpp"

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

private:
    std::vector<ProcType> m_processes;
    std::vector<StateType> m_states;
    std::vector<StateType> m_previous_states;
    int m_cycles;
};

 
