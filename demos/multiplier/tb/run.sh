#!/usr/bin/env bash

set -xe

iverilog -g2012 ../*.sv ./tb.sv -s tb -o run_tb
./run_tb
