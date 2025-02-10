#include "Runtime.hpp"
#include <cassert>
#include <vector>

Circuit::Circuit(int number_of_circuits)
    : m_previous_states({})
    , m_states({})
    , m_cycles(0)
    , m_processes({})
{
    for (int i = 0; i < number_of_circuits; i++) {
        m_states.push_back(StateType());
        m_previous_states.push_back(StateType());
    }

    m_processes = make_processes();
}

void Circuit::apply_input(ApplyInputFunc func)
{
    for (int i = 0; i < m_states.size(); i++) {
        func(&m_states[i], i, m_cycles);
    }
}

static void clone_state(std::vector<StateType>& from, std::vector<StateType>& to) {
    assert(from.size() == to.size());

    for (int i = 0; i < from.size(); i++ ) {
        to[i] = from[i];
    }
}

// TODO: Make this work for more than one state
void Circuit::eval() {
    DiffType diff;
    std::vector<ProcType> ready_queue;

    while (1) {
        // Check which processess need to be run
        state_calculate_diff(&m_previous_states[0], &m_states[0], &diff);

        for (auto& proc : m_processes) {
            for (auto signal_idx : proc.sensitivity) {
                if (diff.is_different[signal_idx]) {
                    ready_queue.push_back(proc);
                    break;
                }
            }
        }

        clone_state(m_states, m_previous_states);

        // If none, we are done
        if (ready_queue.size() == 0) {
            break;
        }

        // Run the processes
        for (auto & proc : ready_queue) {
            proc.function_pointer(&m_states[0], 1);
        }

        ready_queue.clear();

    }
    m_cycles++;

    clone_state(m_states, m_previous_states);
}
