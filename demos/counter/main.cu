#include "fmt/format.h"

#include "codegen/module.hpp"
#include "runtime/Runtime.hpp"
#include "runtime/UserProvided.hpp"

__global__ void cudalator_apply_input(StateType *dut, int cycle, size_t len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;

    dut[tid].i_clk = Bit<1>(cycle % 2);
    if (cycle < 4) {
        dut[tid].i_rst_n = Bit<1>(0);
    } else {
        dut[tid].i_rst_n = Bit<1>(1);
    }
    dut[tid].i_en = Bit<1>(1);
}

int main() {
    Circuit circuit(16);

    circuit.open_vcd("waves.vcd", 3);
    fmt::println("Starting Simulation");
    for (int i = 0; i < 60; i++) {
        circuit.apply_input();
        circuit.eval();
    }
    fmt::println("Simulation Done!");
}
