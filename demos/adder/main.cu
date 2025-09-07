#include <fmt/base.h>

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include "runtime/UserProvided.hpp"

__global__ void cudalator_apply_input(StateType *dut, int cycle, size_t len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;


    int a = tid/2;
    int b = tid - a;

    dut[tid].a = Bit<8>(a % 256);
    dut[tid].b = Bit<8>(b % 256);
}

int main() {
    Circuit circuit(32);

    fmt::println("Starting Simulation");
    for (int i = 0; i < 10; i++) {
        circuit.apply_input();
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
