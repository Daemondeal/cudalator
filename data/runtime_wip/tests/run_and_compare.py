import subprocess
import re
import sys
import os
import shutil

BUILD_DIR = "build"

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Stderr:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def parse_output(output):
    results = {}
    pattern = re.compile(r"TEST\s+([A-Z0-9_]+)\s+Result:\s+([0-9a-fA-F]+)")
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            test_id, value = match.groups()
            results[test_id] = value.lower()
    return results

def main():
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)

    try:
        print("--- 1. Compiling and Running C++ Test ---")
        cpp_source = "special_ops_test.cpp"
        cpp_executable = os.path.join(BUILD_DIR, "special_ops_test")
        # Note the -I.. to find runtime.hpp in the parent directory
        cpp_command = f"g++ -std=c++17 -I.. -o {cpp_executable} {cpp_source} && ./{cpp_executable}"
        cpp_output = run_command(cpp_command)
        cpp_results = parse_output(cpp_output)
        print(f"Found {len(cpp_results)} C++ test results.")

        print("\n--- 2. Compiling and Running Verilog Test ---")
        verilog_source = "special_ops_tb.v"
        verilog_executable = os.path.join(BUILD_DIR, "special_ops_tb")
        verilog_command = f"iverilog -g2012 -o {verilog_executable} {verilog_source} && vvp {verilog_executable}"
        verilog_output = run_command(verilog_command)
        verilog_results = parse_output(verilog_output)
        print(f"Found {len(verilog_results)} Verilog test results.")

        print("\n--- 3. Comparing Results ---")
        all_tests = sorted(list(set(cpp_results.keys()) | set(verilog_results.keys())))
        mismatches = 0

        for test_id in all_tests:
            cpp_res = cpp_results.get(test_id, "MISSING")
            verilog_res = verilog_results.get(test_id, "MISSING")

            if cpp_res == verilog_res:
                print(f"  \033[92m{test_id:<25}\033[0m | MATCH: {cpp_res}")
            else:
                mismatches += 1
                print(f"  \033[91m❌ {test_id:<25}\033[0m | MISMATCH!")
                print(f"     - C++ Output:     {cpp_res}")
                print(f"     - Verilog Output: {verilog_res}")

        print("\n--- Summary ---")
        if mismatches == 0:
            print("\033[92mAll tests passed!\033[0m")
        else:
            print(f"\033[91mFound {mismatches} mismatch(es).\033[0m")
            sys.exit(1)

    finally:
        # This block will run whether the tests pass or fail
        print("\n--- 4. Cleaning up build directory ---")
        if os.path.exists(BUILD_DIR):
            shutil.rmtree(BUILD_DIR)
            print(f"Removed directory: {BUILD_DIR}")

if __name__ == "__main__":
    main()
