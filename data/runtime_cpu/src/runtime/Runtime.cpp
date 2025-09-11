#include "Runtime.hpp"
#include "ChangeType.hpp"
#include "Stats.hpp"
#include "Vcd.hpp"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/os.h>
#include <memory>
#include <string>
#include <vector>

Circuit::Circuit(int number_of_circuits)
    : m_num_circuits(number_of_circuits),
      m_cycles(0),
      m_previous_states({}),
      m_states({}),
      m_stats({}),
      m_processes({}) {
    for (int i = 0; i < number_of_circuits; i++) {
        m_states.push_back(StateType());
        m_previous_states.push_back(StateType());
    }

    m_processes = make_processes();

    m_stats.number_of_circuits = m_num_circuits;
    m_stats.state_array_size = sizeof(StateType) * (m_states.size() + m_previous_states.size());
    m_stats.diff_array_size = sizeof(DiffType);

    first_eval();
}

void Circuit::apply_input(ApplyInputFunc func) {
    for (int i = 0; i < m_states.size(); i++) {
        func(&m_states[0], i, m_cycles);
    }
}

void Circuit::open_vcd(const std::string path, int circuit_idx) {
    if (circuit_idx >= m_num_circuits) {
        fmt::println("ERROR: trying to dump circuit {} when there are only {} circuits", circuit_idx, m_num_circuits);
        // TODO: Handle errors better
        std::exit(1);
    }
    assert(circuit_idx < m_num_circuits);

    auto fp = std::make_unique<fmt::ostream>(fmt::output_file(path));

    m_vcd.emplace(std::move(fp), circuit_idx);
    m_vcd->print_header();
}

void Circuit::dump_to_vcd() {
    if (m_vcd.has_value()) {
        m_vcd->dump(m_states, m_cycles);
    }
}

static void clone_state(std::vector<StateType>& from,
                        std::vector<StateType>& to) {
    assert(from.size() == to.size());

    for (int i = 0; i < from.size(); i++) {
        to[i] = from[i];
    }
}

// TODO: Make this work for more than one state
void Circuit::eval() {
    std::vector<ProcType> ready_queue;

    m_stats.start_counter(PerfEvent::DoDeltaCycle);
    // Iteration
    while (1) {
        DiffType diff{};

        m_stats.start_counter(PerfEvent::DoIteration);

        m_stats.start_counter(PerfEvent::CalculateStateDiff);
        // Check which processess need to be run
        for (int i = 0; i < m_num_circuits; i++) {
            DiffType diff_tmp;
            state_calculate_diff(&m_previous_states[i], &m_states[i], &diff_tmp, 1);

            for (int j = 0 ; j < sizeof(diff.change)/sizeof(ChangeType); j++) {
                // Merge Changes
                ChangeType acc = diff.change[j];
                ChangeType change = diff_tmp.change[j];

                if (acc != change) {
                    switch (acc) {
                    case ChangeType::NoChange:
                        diff.change[j] = change;
                        break;
                    case ChangeType::Change:
                        break;
                    case ChangeType::Posedge:
                        diff.change[j] = ChangeType::Change;
                        break;
                    case ChangeType::Negedge:
                        diff.change[j] = ChangeType::Change;
                        break;
                    }
                }
            }
        }
        m_stats.stop_counter(PerfEvent::CalculateStateDiff);

        m_stats.start_counter(PerfEvent::PopulateReadyQueue);
        for (auto& proc : m_processes) {
            bool should_run = false;
            for (auto [signal_idx, change_type] : proc.sensitivity) {
                switch (change_type) {
                    case ChangeType::Posedge:
                        should_run =
                            (diff.change[signal_idx] == ChangeType::Posedge);
                        break;
                    case ChangeType::Negedge:
                        should_run =
                            (diff.change[signal_idx] == ChangeType::Negedge);
                        break;
                    case ChangeType::Change:
                        should_run =
                            (diff.change[signal_idx] != ChangeType::NoChange);
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
        m_stats.stop_counter(PerfEvent::PopulateReadyQueue);

        m_stats.start_counter(PerfEvent::CloneStates);
        clone_state(m_states, m_previous_states);
        m_stats.stop_counter(PerfEvent::CloneStates);
        // If none, we are done
        if (ready_queue.size() == 0) {
            m_stats.stop_counter(PerfEvent::DoIteration);
            break;
        }
        m_stats.kernels_launched += ready_queue.size();

        m_stats.start_counter(PerfEvent::RunKernels);
        // Run the processes
        for (auto& proc : ready_queue) {
            for (int i = 0; i < m_num_circuits; i++) {
                run_process(&m_states[i], 1, proc.id);
            }
        }
        m_stats.stop_counter(PerfEvent::RunKernels);

        ready_queue.clear();
        m_stats.iterations_done++;

        m_stats.stop_counter(PerfEvent::DoIteration);
    }
    m_cycles++;
    m_stats.delta_times_ran++;
    m_stats.stop_counter(PerfEvent::DoDeltaCycle);

    clone_state(m_states, m_previous_states);

    dump_to_vcd();
}

void Circuit::first_eval() {
    for (auto& proc : m_processes) {
        for (int i = 0; i < m_num_circuits; i++) {
            run_process(&m_states[i], 1, proc.id);
        }
    }
    m_stats.kernels_launched += m_processes.size();

    eval();
    m_cycles--;
}
