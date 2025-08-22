// NOTE: This file is just for debugging purposes, it will be overwritten each time a new project is built.
#pragma once
#include "../runtime/Process.hpp"
#include "../runtime/Bit.hpp"
#include <vector>
#include <cstddef>
#include <fmt/os.h>

struct state_work__double_int_adder
{
  int32_t work__double_int_adder__a_0;
  int32_t work__double_int_adder__b_1;
  int32_t work__double_int_adder__c_2;
  int32_t work__double_int_adder__sum_3;
  int32_t work__double_int_adder__interm_4;
};

struct diff_work__double_int_adder
{
  ChangeType change[5];
};

void state_calculate_diff(state_work__double_int_adder* start, state_work__double_int_adder* end, diff_work__double_int_adder* diffs);

std::vector<Process<state_work__double_int_adder>> make_processes();

using DiffType = diff_work__double_int_adder;
using StateType = state_work__double_int_adder;

void state_vcd_dump_names(fmt::ostream &file);
void state_vcd_dump_values(state_work__double_int_adder *state, int tid, fmt::ostream &file);

