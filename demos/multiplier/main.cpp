#include "fmt/format.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include "runtime/Utils.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>


#ifndef VERBOSE
#define VERBOSE 0
#endif

static void apply_input(StateType* dut, int circuit_idx, int cycle) {
    cu::Random rng(cycle ^ circuit_idx);

    uint32_t a = rng.get();
    uint32_t b = rng.get();

    dut[circuit_idx].i_A = Bit<8>(a % 256);
    dut[circuit_idx].i_B = Bit<8>(b % 256);

#if VERBOSE
    fmt::println("[{:>2}] {} * {} = {} ({})", cycle,
                 dut[circuit_idx].i_A, dut[circuit_idx].i_B,
                 dut[circuit_idx].o_result, expected);

    if (dut[circuit_idx].o_result == expected) {
        fmt::println("- correct");
    } else {
        fmt::println("- incorrect");
    }
#endif
}

int main(int argc, char *argv[]) {
    size_t circuit_number = 16;
    if (argc == 2) {
        circuit_number = atoi(argv[1]);
    }

    Circuit circuit(circuit_number);

    constexpr int MAX = 1000;

    // circuit.open_vcd("waves.vcd", 0);
    fmt::println("Starting Simulation");
    for (int i = 0; i < MAX; i++) {
        cu::progress_display(i, MAX);

        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("");
    fmt::println("Simulation Done!");

    circuit.get_stats().print();
    circuit.get_stats().save_to_json("result.json");
}
