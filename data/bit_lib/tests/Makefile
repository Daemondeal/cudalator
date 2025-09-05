CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -I../src/
IVERILOG = iverilog
IVERILOG_FLAGS = -g2012
VVP = vvp
BUILD_DIR = build
.PHONY: all test clean

all: test

test: $(BUILD_DIR)/cpp_results.txt $(BUILD_DIR)/verilog_results.txt
	@echo "--- Comparing Verilog (Golden) vs. C++ (DUT) ---"
	@python3 generate_tests.py --compare

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cpp_results.txt: $(BUILD_DIR)/runner
	@echo "--- Running C++ Simulation ---"
	@./$(BUILD_DIR)/runner > $(BUILD_DIR)/cpp_results.txt

$(BUILD_DIR)/runner: $(BUILD_DIR)/test_runner.cpp ../src/Bit.hpp ../src/cuda_compat.hpp | $(BUILD_DIR)
	@echo "--- Compiling C++ Test Runner ---"
	@$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/verilog_results.txt: $(BUILD_DIR)/tb
	@echo "--- Running Verilog Simulation ---"
	@$(VVP) $(BUILD_DIR)/tb | grep "Result:" > $(BUILD_DIR)/verilog_results.txt

$(BUILD_DIR)/tb: $(BUILD_DIR)/testbench.v | $(BUILD_DIR)
	@echo "--- Compiling Verilog Testbench ---"
	@$(IVERILOG) $(IVERILOG_FLAGS) -o $@ $<

$(BUILD_DIR)/testbench.v $(BUILD_DIR)/test_runner.cpp: generate_tests.py test_vectors.txt | $(BUILD_DIR)
	@echo "--- Generating Test Files ---"
	@python3 generate_tests.py

clean:
	@echo "--- Cleaning up generated files ---"
	@rm -rf $(BUILD_DIR)
