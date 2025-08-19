#!/usr/bin/env python3
import sys

OP_MAP = {
    "ADD": {
        "verilog": "a + b",
        "cpp": "a + b"
    },
    "SUB": {
        "verilog": "a - b",
        "cpp": "a - b"
    }
}

def format_verilog_value(value, width):
    value = value.strip()
    if value.lower().startswith("0x"):
        # It's a hex value, format as <width>'h<hex_digits>
        hex_digits = value[2:] # Strip the "0x"
        return f"{width}'h{hex_digits}"
    else:
        # It's a decimal value, format as <width>'d<decimal_value>
        return f"{width}'d{value}"

def generate_verilog(lines):
    with open("testbench.v", "w") as f:
        f.write("`timescale 1ns / 1ps\n")
        f.write("module testbench;\n")

        declarations = set()
        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue
            try:
                op, w1, v1, w2, v2, w_res = [s.strip() for s in line.split(',')]
                declarations.add(f"  reg [{int(w1)-1}:0] a_{i};\n")
                declarations.add(f"  reg [{int(w2)-1}:0] b_{i};\n")
                declarations.add(f"  reg [{int(w_res)-1}:0] res_{i};\n")
            except (ValueError, KeyError):
                continue

        for dec in sorted(list(declarations)):
            f.write(dec)
        f.write("\n")

        f.write("  initial begin\n")
        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue

            try:
                op, w1, v1, w2, v2, w_res = [s.strip() for s in line.split(',')]

                a_name = f"a_{i}"
                b_name = f"b_{i}"
                res_name = f"res_{i}"

                verilog_expr = OP_MAP[op]["verilog"]

                f.write(f"    {a_name} = {format_verilog_value(v1, w1)};\n")
                f.write(f"    {b_name} = {format_verilog_value(v2, w2)};\n")

                f.write(f"    {res_name} = {verilog_expr.replace('a', a_name).replace('b', b_name)};\n")

                f.write(f'    $display("Test: {line.strip()} -> Result: %h", {res_name});\n\n')

            except (ValueError, KeyError) as e:
                print(f"Skipping malformed line in test_vectors.txt: {line.strip()} -> {e}")

        f.write("    #1 $finish;\n")
        f.write("  end\nendmodule\n")

def generate_cpp(lines):
    """Generates the C++ test runner."""
    with open("test_runner.cpp", "w") as f:
        f.write('#include "runtime.hpp"\n')
        f.write('#include <iostream>\n\n')
        f.write('int main() {\n')

        for line in lines:
            if line.startswith('#') or not line.strip():
                continue

            try:
                op, w1, v1, w2, v2, w_res = [s.strip() for s in line.split(',')]

                # For C++, we need to wrap hex strings in quotes for the string constructor
                cpp_v1 = f'"{v1}"' if v1.lower().startswith("0x") else v1
                cpp_v2 = f'"{v2}"' if v2.lower().startswith("0x") else v2

                cpp_expr = OP_MAP[op]["cpp"]

                f.write("    {\n")
                f.write(f"        Bit<{w1}> a({cpp_v1});\n")
                f.write(f"        Bit<{w2}> b({cpp_v2});\n")
                f.write(f"        Bit<{w_res}> result = {cpp_expr};\n")
                f.write(f'        std::cout << "Test: {line.strip()} -> Result: " << result.to_string() << std::endl;\n')
                f.write("    }\n")

            except (ValueError, KeyError) as e:
                print(f"Skipping malformed line in test_vectors.txt: {line.strip()} -> {e}")

        f.write('    return 0;\n}\n')

if __name__ == "__main__":
    try:
        with open("test_vectors.txt", "r") as f:
            lines = f.readlines()
        generate_verilog(lines)
        generate_cpp(lines)
        print("Successfully generated testbench.v and test_runner.cpp")
    except FileNotFoundError:
        print("Error: test_vectors.txt not found. Please create it.")
    except Exception as e:
        print(f"An error occurred: {e}")
