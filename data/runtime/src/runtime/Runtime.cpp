#include "Runtime.hpp"
#include "ChangeType.hpp"
#include "Vcd.hpp"
#include <cassert>
#include <fmt/format.h>
#include <fmt/os.h>
#include <memory>
#include <string>
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

    first_eval();
}

void Circuit::apply_input(ApplyInputFunc func)
{
    for (int i = 0; i < m_states.size(); i++) {
        func(&m_states[i], i, m_cycles);
    }
}

void Circuit::open_vcd(const std::string path, int circuit_idx) {
    auto fp = std::make_unique<fmt::ostream>(fmt::output_file(path));

    m_vcd.emplace(std::move(fp), circuit_idx);
    m_vcd->print_header();
}


void Circuit::dump_to_vcd() {
    if (m_vcd.has_value()) {
        m_vcd->dump(m_states, m_cycles);
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
            bool should_run = false;
            for (auto [signal_idx, change_type]  : proc.sensitivity) {
                switch (change_type) {
                case ChangeType::Posedge:
                    should_run = (diff.change[signal_idx] == ChangeType::Posedge);
                    break;
                case ChangeType::Negedge:
                    should_run = (diff.change[signal_idx] == ChangeType::Negedge);
                    break;
                case ChangeType::Change:
                    should_run = (diff.change[signal_idx] != ChangeType::NoChange);
                    break;
                case ChangeType::NoChange:
                    assert(false);
                    break;
                }
                if (should_run) {
                    break;
                }
            }

            if (should_run) {
                ready_queue.push_back(proc);
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

    dump_to_vcd();
}


void Circuit::first_eval() {
    for (auto & proc : m_processes) {
        proc.function_pointer(&m_states[0], 1);
    }

    eval();
    m_cycles--;
}
