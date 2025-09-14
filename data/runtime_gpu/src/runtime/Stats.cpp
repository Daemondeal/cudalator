#include "Stats.hpp"
#include <cassert>
#include <chrono>
#include <fmt/base.h>
#include <fmt/os.h>
#include <vector>

static std::string format_time(double seconds) {
    double t = seconds;
    const char* unit = nullptr;

    if (t >= 1.0) {
        unit = "s";
    } else if (t >= 1e-3) {
        t *= 1e3;
        unit = "ms";
    } else {
        t *= 1e6;
        unit = "µs";
    }

    return fmt::format("{:.3f} {}", t, unit);
}

std::string format_bytes(uint64_t bytes) {
    const char* units[] = { "B", "kB", "MB", "GB", "TB" };
    double value = static_cast<double>(bytes);
    int unit_index = 0;

    while (value >= 1000.0 && unit_index < 4) {
        value /= 1000.0;
        ++unit_index;
    }

    return fmt::format("{:.3f} {}", value, units[unit_index]);
}

#define STAT_NUMERIC(name, value)    stats.emplace_back(name, fmt::format("{}", value), value)
#define STAT_PERCENTAGE(name, value) stats.emplace_back(name, fmt::format("{:.2f} %", value), value)
#define STAT_TIME(name, time)        stats.emplace_back(name, format_time(time), time);
#define STAT_MEMORY_SIZE(name, size) stats.emplace_back(name, format_bytes(size), size);


std::vector<StatInfo> Stats::gather_stats() const {
    std::vector<StatInfo> stats;

    double iters = static_cast<double>(iterations_done);

    auto pc_state_diff = get_counter(PerfEvent::CalculateStateDiff);
    auto pc_memcopy_state = get_counter(PerfEvent::MemcopyStateForPopulating);
    auto pc_ready_queue = get_counter(PerfEvent::PopulateReadyQueue);
    auto pc_clone_states = get_counter(PerfEvent::CloneStates);
    auto pc_run_kernels = get_counter(PerfEvent::RunKernels);
    auto pc_do_iterations = get_counter(PerfEvent::DoIteration);
    auto pc_do_delta_cycle = get_counter(PerfEvent::DoDeltaCycle);

    STAT_NUMERIC("Number of Circuits",             number_of_circuits);
    STAT_NUMERIC("Kernels Launched",               kernels_launched);
    STAT_NUMERIC("Iterations Done",                iterations_done);
    STAT_NUMERIC("Delta Times Ran",                delta_times_ran);
    STAT_NUMERIC("Iterations per Delta",           iters / static_cast<double>(delta_times_ran));

    STAT_MEMORY_SIZE("State Array Size",           state_array_size);
    STAT_MEMORY_SIZE("Diff Array Size",            diff_array_size);

    STAT_TIME   ("Total Time to Diff States",      pc_state_diff.total_time);
    STAT_TIME   ("Total Time to Memcopy State",    pc_memcopy_state.total_time);
    STAT_TIME   ("Total Time to Populate Queue",   pc_ready_queue.total_time);
    STAT_TIME   ("Total Time to Clone States",     pc_clone_states.total_time);
    STAT_TIME   ("Total Time to Run Kernels",      pc_run_kernels.total_time);
    STAT_TIME   ("Total Time to Do an Iteration",  pc_do_iterations.total_time);
    STAT_TIME   ("Total Time to Do a Delta Cycle", pc_do_delta_cycle.total_time);

    STAT_TIME   ("Average Time to Diff",             pc_state_diff.total_time / pc_state_diff.count);
    STAT_TIME   ("Average Time to Memcopy State",    pc_memcopy_state.total_time / pc_clone_states.count);
    STAT_TIME   ("Average Time to Populate Queue",   pc_ready_queue.total_time / pc_clone_states.count);
    STAT_TIME   ("Average Time to Clone States",     pc_clone_states.total_time / pc_clone_states.count);
    STAT_TIME   ("Average Time to Run Kernels",      pc_run_kernels.total_time / pc_run_kernels.count);
    STAT_TIME   ("Average Time to Do an Iteration",  pc_do_iterations.total_time / pc_do_iterations.count);
    STAT_TIME   ("Average Time to Do a Delta Cycle", pc_do_delta_cycle.total_time / pc_do_delta_cycle.count);

    STAT_PERCENTAGE("Pct. iter time spent on state diff",             100 * pc_state_diff.total_time / pc_do_iterations.total_time );
    STAT_PERCENTAGE("Pct. iter time spent on memcopying state",       100 * pc_memcopy_state.total_time / pc_do_iterations.total_time );
    STAT_PERCENTAGE("Pct. iter time spent on populating ready queue", 100 * pc_ready_queue.total_time / pc_do_iterations.total_time );
    STAT_PERCENTAGE("Pct. iter time spent on running kernels ",       100 * pc_run_kernels.total_time / pc_do_iterations.total_time );
    STAT_PERCENTAGE("Pct. iter time spent on state cloning per iter", 100 * pc_clone_states.total_time / pc_do_iterations.total_time );

    return stats;
}


void Stats::print() const {
    auto stats = gather_stats();

    size_t max_name_len = 0;
    for (const auto& s : stats) {
        max_name_len = std::max(max_name_len, s.name.size());
    }

    for (const auto& s : stats) {
        fmt::print("{:<{}} {}\n", s.name + ":", max_name_len + 1, s.value);
    }
}

void Stats::save_to_json(std::string path_json) const {
    auto stats = gather_stats();

    auto fp = fmt::output_file(path_json);
    fp.print("{{\n");

    for (size_t i = 0; i < stats.size(); ++i) {
        const auto& s = stats[i];
        if (i + 1 == stats.size()) {
            fp.print("    \"{}\": {}\n", s.name, s.raw_value);
        } else {
            fp.print("    \"{}\": {},\n", s.name, s.raw_value);
        }
    }


    fp.print("}}\n");

}

void Stats::start_counter(PerfEvent event) {
    auto &counter = perf_counters[static_cast<size_t>(event)];
    assert(!counter.is_started);

    counter.is_started = true;
    counter.start_time = std::chrono::high_resolution_clock::now();
}

void Stats::stop_counter(PerfEvent event) {
    auto &counter = perf_counters[static_cast<size_t>(event)];
    assert(counter.is_started);

    counter.is_started = false;
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff = now - counter.start_time;

    counter.total_time += diff.count();
    counter.count += 1;
}

const PerfCounter& Stats::get_counter(PerfEvent event) const {
    return perf_counters[static_cast<size_t>(event)];
}
