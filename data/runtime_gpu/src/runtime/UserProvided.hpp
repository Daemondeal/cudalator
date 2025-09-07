#pragma once

#include "Runtime.hpp"

__global__ void cudalator_apply_input(StateType *dut, int cycle, size_t len);
