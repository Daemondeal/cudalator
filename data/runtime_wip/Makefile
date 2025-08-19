CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
IVERILOG = iverilog
VVP = vvp

all: test

test: verilog_results.txt cpp_results.txt
	@echo "--- Comparing Verilog (Golden) vs. C++ (DUT) ---"
	@python3 generate_tests.py --compare

verbose_test: verilog_results.txt cpp_results.txt
	@echo "--- Verbose Comparison ---"
	@echo "\n--- Verilog Output (Golden): ---"
	@cat verilog_results.txt
	@echo "\n--- C++ Output (DUT): ---"
	@cat cpp_results.txt
	@echo "\n--- Diff (empty if tests pass): ---"
	@diff -u verilog_results.txt cpp_results.txt || true

verilog_results.txt: tb
	@echo "--- Running Verilog Simulation ---"
	@$(VVP) tb | grep "Result:" > verilog_results.txt

cpp_results.txt: runner
	@echo "--- Running C++ Simulation ---"
	@./runner > cpp_results.txt

tb: testbench.v
	@echo "--- Compiling Verilog Testbench ---"
	@$(IVERILOG) -o tb testbench.v

runner: test_runner.cpp runtime.hpp
	@echo "--- Compiling C++ Test Runner ---"
	@$(CXX) $(CXXFLAGS) -o runner test_runner.cpp

testbench.v test_runner.cpp: generate_tests.py test_vectors.txt
	@echo "--- Generating Test Files ---"
	@python3 generate_tests.py

clean:
	@echo "--- Cleaning up generated files ---"
	@rm -f tb runner testbench.v test_runner.cpp verilog_results.txt cpp_results.txt

.PHONY: all test clean
