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

    int seed = cycle ^ tid;
    xorshift32_state rng;
    rng.a = seed;
    xorshift32(&rng);


    if (cycle % 2 == 0) {
        dut[tid].CK = 0;
        return;
    }

    #define DRIVE(input) dut[tid].input = (xorshift32(&rng) % 2)
    dut[tid].CK = 1;

    DRIVE(n3065gat);
    DRIVE(n3066gat);
    DRIVE(n3067gat);
    DRIVE(n3068gat);
    DRIVE(n3069gat);
    DRIVE(n3070gat);
    DRIVE(n3071gat);
    DRIVE(n3072gat);
    DRIVE(n3073gat);
    DRIVE(n3074gat);
    DRIVE(n3075gat);
    DRIVE(n3076gat);
    DRIVE(n3077gat);
    DRIVE(n3078gat);
    DRIVE(n3079gat);
    DRIVE(n3080gat);
    DRIVE(n3081gat);
    DRIVE(n3082gat);
    DRIVE(n3083gat);
    DRIVE(n3084gat);
    DRIVE(n3085gat);
    DRIVE(n3086gat);
    DRIVE(n3087gat);
    DRIVE(n3088gat);
    DRIVE(n3089gat);
    DRIVE(n3090gat);
    DRIVE(n3091gat);
    DRIVE(n3092gat);
    DRIVE(n3093gat);
    DRIVE(n3094gat);
    DRIVE(n3095gat);
    DRIVE(n3097gat);
    DRIVE(n3098gat);
    DRIVE(n3099gat);
    DRIVE(n3100gat);
}

int main(int argc, char *argv[]) {
    size_t circuit_number = 256;
    if (argc == 2) {
        circuit_number = atoi(argv[1]);
    }

    Circuit circuit(circuit_number);

    constexpr int MAX = 1000;
    constexpr int STEP = std::max(MAX / 1000, 1);

    // circuit.open_vcd("waves.vcd", 3);
    fmt::println("Starting Simulation");
    for (int i = 0; i < MAX; i++) {
        if (i % STEP == 0) {
            int steps = (20 * (i+1)) / MAX;

            fmt::print("\r[");
            for (int j = 0; j < steps; j++) {
                fmt::print("#");
            }

            for (int j = steps; j < 20; j++) {
                fmt::print(".");
            }
            fmt::print("] {}/{}", i+1, MAX);
            fflush(stdout);
        }


        circuit.apply_input();
        circuit.eval();
    }
    fmt::println("\nSimulation Done!");

    circuit.get_stats().print();
    circuit.get_stats().save_to_json("result.json");
}
