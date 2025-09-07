#include <fmt/base.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {
    fmt::println(
        "[{:>2}] {} + {} = {} (expected {})",
        cycle,
        dut[circuit_idx].work__int_adder__a_0,
        dut[circuit_idx].work__int_adder__b_1,
        dut[circuit_idx].work__int_adder__sum_2,
        dut[circuit_idx].work__int_adder__a_0 + dut[circuit_idx].work__int_adder__b_1
    );

    dut[circuit_idx].work__int_adder__a_0 = cycle;
    dut[circuit_idx].work__int_adder__b_1 = cycle * 2;
}

int main() {
    Circuit circuit(1);

    fmt::println("Starting Simulation");
    for (int i = 0; i < 10; i++) {
        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
