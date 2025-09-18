#!/usr/bin/env bash

set -e

echo "### BUILDING ###"
verilator --cc ../s5378.sv --exe --build verilator.cpp --top-module s5378

echo
echo "### RUNNING ###"
time ./obj_dir/Vs5378 $1
