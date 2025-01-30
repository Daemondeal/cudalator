CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra

# Verilog
VLOG := iverilog
VLOG_RUN := vvp

all: runtime_test verilog_test

# C++ test program rule
runtime_test: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "\nRunning runtime test..."
	@./$@

# Verilog test rule
verilog_test: bit_ops_tb.v
	$(VLOG) -o simv $<
	@echo "\nRunning Verilog test..."
	@$(VLOG_RUN) simv
	@echo "Verilog results:"
	@cat results.txt

clean:
	rm -f runtime_test simv results.txt *.vcd

.PHONY: all clean runtime_test verilog_test
