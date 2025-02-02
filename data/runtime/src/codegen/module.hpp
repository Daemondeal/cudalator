// NOTE: This file is just for debugging purposes, it will be overwritten each time a new project is built.

#pragma once
#include "../runtime/Process.hpp"
#include "../runtime/Bit.hpp"
#include <vector>
#include <cstddef>

struct state_work__adder
{
  Bit<8> work__adder__a;
  Bit<8> work__adder__b;
  Bit<8> work__adder__c;
  Bit<1> work__adder__cout;
  Bit<9> work__adder__full_sum;
};

struct diff_work__adder
{
  bool is_different[5];
};

void state_calculate_diff(state_work__adder* start, state_work__adder* end, diff_work__adder* diffs);

std::vector<Process<state_work__adder>> make_processes();

using DiffType = diff_work__adder;
using StateType = state_work__adder;

