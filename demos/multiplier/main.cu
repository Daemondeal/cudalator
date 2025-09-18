#include <fmt/base.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include "runtime/UserProvided.hpp"
#include "runtime/Utils.hpp"


__global__ void cudalator_apply_input(StateType *dut, int cycle, size_t len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;

    cu::Random random(cycle ^ tid);
    uint32_t a = random.get();
    uint32_t b = random.get();

    dut[tid].i_A = Bit<8>(a % 256);
    dut[tid].i_B = Bit<8>(b % 256);
}

int main(int argc, char *argv[]) {
    size_t circuit_number = 256;
    if (argc == 2) {
        circuit_number = atoi(argv[1]);
    }

    Circuit circuit(circuit_number);

    constexpr int MAX = 1000;

    circuit.open_vcd("waves.vcd", 72);
    fmt::println("Starting Simulation");
    for (int i = 0; i < MAX; i++) {
        cu::progress_display(i, MAX);

        circuit.apply_input();
        circuit.eval();
    }
    fmt::println("");
    fmt::println("Simulation Done!");

    circuit.get_stats().print();
    circuit.get_stats().save_to_json("result.json");
}
