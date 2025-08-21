#include <fmt/core.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {
    fmt::println(
        "[{:>2}] {} + {} + {} = {} {}",
        cycle,
        dut[circuit_idx].a,
        dut[circuit_idx].b,
        dut[circuit_idx].cin,
        dut[circuit_idx].s,
        dut[circuit_idx].cout
    );

    static int64_t random = 81273981;

    random ^= random << 7;
    random ^= random >> 9;
    dut[circuit_idx].a = random & 1;

    random ^= random << 7;
    random ^= random >> 9;
    dut[circuit_idx].b = random & 1;

    random ^= random << 7;
    random ^= random >> 9;
    dut[circuit_idx].cin = random & 1;
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
