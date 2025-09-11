#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct StatInfo {
    std::string name;
    std::string value;
    float raw_value;

    StatInfo(std::string name, std::string value, float raw_value)
        : name(name), value(value), raw_value(raw_value) {}
};

enum class PerfEvent {
    CalculateStateDiff,
    PopulateReadyQueue,
    CloneStates,
    RunKernels,

    DoIteration,
    DoDeltaCycle,

    Count
};

struct PerfCounter {
    double total_time;
    uint64_t count;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    bool is_started;
};

struct Stats {
    uint64_t kernels_launched;
    uint64_t iterations_done;
    uint64_t delta_times_ran;

    uint64_t state_array_size;
    uint64_t diff_array_size;
    uint64_t number_of_circuits;

    std::array<PerfCounter, static_cast<size_t>(PerfEvent::Count)> perf_counters;

    void start_counter(PerfEvent event);
    void stop_counter(PerfEvent event);
    const PerfCounter& get_counter(PerfEvent event) const;



    void print() const;
    void save_to_json(std::string path_json) const;

private:
    std::vector<StatInfo> gather_stats() const;
};
