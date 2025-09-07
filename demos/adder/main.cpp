#include <fmt/base.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {
    int a = circuit_idx/2;
    int b = circuit_idx - a;

    dut[circuit_idx].a = Bit<8>(a);
    dut[circuit_idx].b = Bit<8>(b);
}

int main() {
    Circuit circuit(8);

    circuit.open_vcd("waves.vcd", 3);
    fmt::println("Starting Simulation");
    for (int i = 0; i < 10; i++) {
        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
