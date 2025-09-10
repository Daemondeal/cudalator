#include <fmt/base.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include "runtime/UserProvided.hpp"

struct xorshift32_state {
    uint32_t a;
};

/* The state must be initialized to non-zero */
__device__ uint32_t xorshift32(xorshift32_state *state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state->a;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state->a = x;
}


__global__ void cudalator_apply_input(StateType *dut, int cycle, size_t len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;

    int seed = cycle % tid;
    xorshift32_state rng;
    rng.a = seed;
    xorshift32(&rng);
    uint32_t a = xorshift32(&rng);
    uint32_t b = xorshift32(&rng);

    dut[tid].i_A = Bit<8>(a % 256);
    dut[tid].i_B = Bit<8>(b % 256);
}

int main() {
    Circuit circuit(64);

    circuit.open_vcd("waves.vcd", 3);
    fmt::println("Starting Simulation");
    for (int i = 0; i < 20; i++) {
        circuit.apply_input();
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
