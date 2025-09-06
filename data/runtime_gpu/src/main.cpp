#include "fmt/format.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {}

int main() {
    Circuit circuit(1);

    circuit.open_vcd("waves.vcd", 0);

    fmt::println("Starting GPU Simulation");
    for (int i = 0; i < 60; i++) {
        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("Simulation Done!");

    return 0;
}
