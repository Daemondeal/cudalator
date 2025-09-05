import subprocess
import sys
import os
import shutil

BUILD_DIR = "build_adder"

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Stderr:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def parse_output(output):
    results = []
    for line in output.splitlines():
        if line.startswith("TEST"):
            results.append(line.strip())
    return results

def main():
    # Ensure the build directory exists and is clean
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)

    # Create a temporary Verilog file for the DUT
    verilog_dut_path = os.path.join(BUILD_DIR, "adder.v")
    with open(verilog_dut_path, "w") as f:
        f.write("""
module adder (
    input logic [7:0] a,
    input logic [7:0] b,
    output logic [7:0] c,
    output logic cout
);
    logic [8:0] full_sum;
    assign full_sum = a + b;
    assign c = full_sum[7:0];
    assign cout = full_sum[8];
endmodule
        """)

    try:
        print("--- 1. Compiling and Running C++ Test ---")
        cpp_executable = os.path.join(BUILD_DIR, "adder_runner")
        cpp_command = f"g++ -std=c++17 -I.. -o {cpp_executable} test_adder_runner.cpp && ./{cpp_executable}"
        cpp_output = run_command(cpp_command)
        cpp_results = parse_output(cpp_output)
        print(f"Found {len(cpp_results)} C++ test results.")

        print("\n--- 2. Compiling and Running Verilog Test ---")
        verilog_executable = os.path.join(BUILD_DIR, "tb_adder")
        verilog_command = f"iverilog -g2012 -o {verilog_executable} tb_adder.v && vvp {verilog_executable}"
        verilog_output = run_command(verilog_command)
        verilog_results = parse_output(verilog_output)
        print(f"Found {len(verilog_results)} Verilog test results.")

        print("\n--- 3. Comparing Results ---")
        mismatches = 0
        for i, (cpp_line, verilog_line) in enumerate(zip(cpp_results, verilog_results)):
            cpp_line_normalized = "".join(cpp_line.split())
            verilog_line_normalized = "".join(verilog_line.split())

            if cpp_line_normalized == verilog_line_normalized:
                print(f"  \033[92m Test {i+1} PASSED\033[0m: {cpp_line}")
            else:
                mismatches += 1
                print(f"  \033[91m❌ Test {i+1} FAILED\033[0m:")
                print(f"     - C++ Output:     {cpp_line}")
                print(f"     - Verilog Output: {verilog_line}")

        print("\n--- Summary ---")
        if mismatches == 0:
            print("\033[92m All adder circuit tests passed!\033[0m")
        else:
            print(f"\033[91m Found {mismatches} mismatch(es).\033[0m")
            sys.exit(1)

    finally:
        print("\n--- 4. Cleaning up build directory ---")
        if os.path.exists(BUILD_DIR):
            shutil.rmtree(BUILD_DIR)
            print(f"Removed directory: {BUILD_DIR}")

if __name__ == "__main__":
    main()
