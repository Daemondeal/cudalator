#include "fmt/format.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include <algorithm>
#include <cstdio>

static void apply_input(StateType* dut, int circuit_idx, int cycle) {

    Bit<16> expected = dut[circuit_idx].i_A * dut[circuit_idx].i_B;

    // fmt::println("[{:>2}] {} * {} = {} ({})", cycle,
    //              dut[circuit_idx].i_A, dut[circuit_idx].i_B,
    //              dut[circuit_idx].o_result, expected);

    dut[circuit_idx].i_A = (cycle) % 256;
    dut[circuit_idx].i_B = (cycle + 1) % 256;

    // if (dut[circuit_idx].o_result == expected) {
    //     fmt::println("- correct");
    // } else {
    //     fmt::println("- incorrect");
    // }
}

int main() {
    Circuit circuit(256);

    constexpr int MAX = 100;
    constexpr int STEP = std::max(MAX / 1000, 1);

    // circuit.open_vcd("waves.vcd", 0);
    fmt::println("Starting Simulation");
    for (int i = 0; i < MAX; i++) {
        if (i % STEP == 0) {
            int steps = (20 * (i+1)) / MAX;

            fmt::print("\r[");
            for (int i = 0; i < steps; i++) {
                fmt::print("#");
            }

            for (int i = steps; i < 20; i++) {
                fmt::print(".");
            }
            fmt::print("] {}/{}", i+1, MAX);
            fflush(stdout);
        }


        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("");
    fmt::println("Simulation Done!");

    circuit.get_stats().print();
}
