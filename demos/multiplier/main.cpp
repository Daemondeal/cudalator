#include "fmt/format.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {

    Bit<16> expected = dut[circuit_idx].i_A * dut[circuit_idx].i_B;

    fmt::println("[{:>2}] {} * {} = {} ({})", cycle,
                 dut[circuit_idx].i_A, dut[circuit_idx].i_B,
                 dut[circuit_idx].o_result, expected);

    dut[circuit_idx].i_A = (cycle) % 256;
    dut[circuit_idx].i_B = (cycle + 1) % 256;

    if (dut[circuit_idx].o_result == expected) {
        fmt::println("- correct");
    } else {
        fmt::println("- incorrect");
    }
}

int main() {
    Circuit circuit(1);

    circuit.open_vcd("waves.vcd", 0);
    fmt::println("Starting Simulation");
    for (int i = 0; i < 60; i++) {
        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
