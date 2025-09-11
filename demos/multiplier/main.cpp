#include "fmt/format.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include <algorithm>
#include <cstdio>

struct xorshift32_state {
    uint32_t a;
};

/* The state must be initialized to non-zero */
uint32_t xorshift32(xorshift32_state *state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state->a;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state->a = x;
}

static void apply_input(StateType* dut, int circuit_idx, int cycle) {

    // Bit<16> expected = dut[circuit_idx].i_A * dut[circuit_idx].i_B;

    int seed = cycle ^ circuit_idx;
    xorshift32_state rng;
    rng.a = seed;
    xorshift32(&rng);
    uint32_t a = xorshift32(&rng);
    uint32_t b = xorshift32(&rng);

    dut[circuit_idx].i_A = Bit<8>(a % 256);
    dut[circuit_idx].i_B = Bit<8>(b % 256);

    // fmt::println("[{:>2}] {} * {} = {} ({})", cycle,
    //              dut[circuit_idx].i_A, dut[circuit_idx].i_B,
    //              dut[circuit_idx].o_result, expected);

    // if (dut[circuit_idx].o_result == expected) {
    //     fmt::println("- correct");
    // } else {
    //     fmt::println("- incorrect");
    // }
}

int main() {
    Circuit circuit(256);

    constexpr int MAX = 1000;
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
