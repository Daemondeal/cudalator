import subprocess
import re
import sys

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Stderr:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def parse_output(output):
    results = {}
    # finds lines tike TEST ID Result: sadas
    pattern = re.compile(r"TEST\s+([A-Z0-9_]+)\s+Result:\s+([0-9a-fA-F]+)")
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            test_id, value = match.groups()
            results[test_id] = value.lower() # Normalize to lowercase
    return results

def main():
    print("--- Compiling and Running C++ Test ---")
    cpp_command = "g++ -std=c++17 -o special_ops_test special_ops_test.cpp && ./special_ops_test"
    cpp_output = run_command(cpp_command)
    cpp_results = parse_output(cpp_output)
    print(f"Found {len(cpp_results)} C++ test results.")

    print("\n--- Compiling and Running Verilog Test ---")
    verilog_command = "iverilog -g2012 -o special_ops_tb special_ops_tb.v && vvp special_ops_tb"
    verilog_output = run_command(verilog_command)
    verilog_results = parse_output(verilog_output)
    print(f"Found {len(verilog_results)} Verilog test results.")

    print("\n--- Comparing Results ---")
    all_tests = sorted(list(set(cpp_results.keys()) | set(verilog_results.keys())))
    mismatches = 0

    for test_id in all_tests:
        cpp_res = cpp_results.get(test_id, "MISSING")
        verilog_res = verilog_results.get(test_id, "MISSING")

        if cpp_res == verilog_res:
            print(f"    {test_id:<20} | MATCH: {cpp_res}")
        else:
            mismatches += 1
            print(f"  ❌ {test_id:<20} | MISMATCH!")
            print(f"     - C++ Output:     {cpp_res}")
            print(f"     - Verilog Output: {verilog_res}")

    print("\n--- Summary ---")
    if mismatches == 0:
        print("All tests passed!")
    else:
        print(f"Found {mismatches} mismatch(es).")
        sys.exit(1)

if __name__ == "__main__":
    main()
