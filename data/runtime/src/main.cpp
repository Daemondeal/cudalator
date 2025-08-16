#include "fmt/core.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"


static void apply_input(StateType *dut, int circuit_idx, int cycle) {
    // Apply inputs to your DUT
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
