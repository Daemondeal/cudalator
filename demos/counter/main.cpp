#include <fmt/core.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {
    fmt::println(
        "[{:>2}] clk: {} rst_n: {} cnt: {} tc: {}",
        cycle,
        dut[circuit_idx].i_clk,
        dut[circuit_idx].i_rst_n,
        dut[circuit_idx].o_cnt,
        dut[circuit_idx].o_tc
    );

    dut[circuit_idx].i_clk = Bit<1>(cycle % 2);
    if (cycle < 4) {
        dut[circuit_idx].i_rst_n = Bit<1>(0);
    } else {
        dut[circuit_idx].i_rst_n = Bit<1>(1);
    }
    dut[circuit_idx].i_en = Bit<1>(1);
}

int main() {
    Circuit circuit(1);

    fmt::println("Starting Simulation");
    for (int i = 0; i < 60; i++) {
        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
