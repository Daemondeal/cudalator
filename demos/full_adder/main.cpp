#include <fmt/base.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"

static void apply_input(StateType* dut, int circuit_idx, int cycle) {
    uint64_t a  = (uint64_t)dut[circuit_idx].a;
    uint64_t b  = (uint64_t)dut[circuit_idx].b;
    uint64_t cin = (uint64_t)dut[circuit_idx].cin;
    uint64_t sum = ((uint64_t)dut[circuit_idx].s) + ((uint64_t)(dut[circuit_idx].cout) << 1);
    uint64_t sum_expected = a + b + cin;

    fmt::println(
        "[{:>2}] {} + {} + {} = {} (expect {})",
        cycle,
        a, b, cin,
        sum, sum_expected
    );
    if (sum != sum_expected) {
        fmt::println("-  WRONG");
    }

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

    circuit.open_vcd("waves.vcd", 0);
    fmt::println("Starting Simulation");
    for (int i = 0; i < 60; i++) {
        circuit.apply_input(apply_input);
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
