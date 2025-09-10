#!/usr/bin/env bash

iverilog -g2012 ../*.sv ./tb.sv -s tb -o run_tb
./run_tb
