#!/usr/bin/env python3
import sys
import re
# The "type" is needed to distinguish unary vs. binary ops.
OP_MAP = {
    "ADD":    {"type": "binary", "verilog": "a + b", "cpp": "a + b"},
    "SUB":    {"type": "binary", "verilog": "a - b", "cpp": "a - b"},
    "MUL":    {"type": "binary", "verilog": "a * b", "cpp": "a * b"},
    "LT":     {"type": "binary", "verilog": "a < b", "cpp": "a < b"},
    "GT":     {"type": "binary", "verilog": "a > b", "cpp": "a > b"},
    "LTE":    {"type": "binary", "verilog": "a <= b","cpp": "a <= b"},
    "GTE":    {"type": "binary", "verilog": "a >= b","cpp": "a >= b"},
    "LSHIFT": {"type": "binary", "verilog": "a << b","cpp": "a << b"},
    "RSHIFT": {"type": "binary", "verilog": "a >> b","cpp": "a >> b"},
    "LNOT":   {"type": "unary",  "verilog": "!a",    "cpp": "!a"},
    "DIV": {"type": "binary", "verilog": "a / b", "cpp": "a / b"},
    "MOD": {"type": "binary", "verilog": "a % b", "cpp": "a % b"},
    "AND": {"type": "binary", "verilog": "a & b","cpp": "a & b"},
}

def format_verilog_value(value, width):
    """Formats a value string into a correctly sized Verilog literal."""
    value = value.strip().upper().replace("ULL", "")
    if value.startswith("0X"):
        hex_digits = value[2:]
        return f"{width}'h{hex_digits}"
    else:
        return f"{width}'d{value}"

def format_cpp_value(value):
    """Formats a value for C++, wrapping hex strings in quotes."""
    value = value.strip()
    if value.lower().startswith("0x"):
        return f'"{value}"'
    else:
        return value

def generate_verilog(lines):
    """Generates the Verilog testbench (the 'golden model')."""
    with open("testbench.v", "w") as f:
        f.write("`timescale 1ns / 1ps\n")
        f.write("module testbench;\n")

        declarations = set()
        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip(): continue
            try:
                op, w1, v1, w2, v2, w_res = [s.strip() for s in line.split(',')]
                op_type = OP_MAP[op]["type"]

                declarations.add(f"  reg [{int(w1)-1}:0] a_{i};\n")
                if op_type == "binary":
                    declarations.add(f"  reg [{int(w2)-1}:0] b_{i};\n")
                declarations.add(f"  reg [{int(w_res)-1}:0] res_{i};\n")
            except (ValueError, KeyError): continue

        for dec in sorted(list(declarations)): f.write(dec)
        f.write("\n")

        f.write("  initial begin\n")
        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip(): continue
            try:
                op, w1, v1, w2, v2, w_res = [s.strip() for s in line.split(',')]
                op_info = OP_MAP[op]

                a_name = f"a_{i}"
                res_name = f"res_{i}"
                verilog_expr = op_info["verilog"]

                f.write(f"    {a_name} = {format_verilog_value(v1, w1)};\n")

                if op_info["type"] == "binary":
                    b_name = f"b_{i}"
                    f.write(f"    {b_name} = {format_verilog_value(v2, w2)};\n")
                    f.write(f"    {res_name} = {verilog_expr.replace('a', a_name).replace('b', b_name)};\n")
                else: # Unary
                    f.write(f"    {res_name} = {verilog_expr.replace('a', a_name)};\n")

                f.write(f'    $display("Test: {line.strip()} -> Result: %h", {res_name});\n\n')
            except (ValueError, KeyError) as e:
                print(f"Skipping malformed line: {line.strip()} -> {e}")

        f.write("    #1 $finish;\n")
        f.write("  end\nendmodule\n")

def generate_cpp(lines):
    """Generates the C++ test runner for ALL cases."""
    with open("test_runner.cpp", "w") as f:
        f.write('#include "runtime.hpp"\n')
        f.write('#include <iostream>\n\n')
        f.write('int main() {\n')

        for line in lines:
            if line.startswith('#') or not line.strip(): continue
            try:
                # This logic is now the same for all operators
                op, w1, v1, w2, v2, w_res = [s.strip() for s in line.split(',')]
                op_info = OP_MAP[op]
                cpp_v1 = format_cpp_value(v1)
                cpp_expr = op_info["cpp"]

                f.write("    {\n")
                f.write(f"        Bit<{w1}> a({cpp_v1});\n")

                if op_info["type"] == "binary":
                    cpp_v2 = format_cpp_value(v2)
                    f.write(f"        Bit<{w2}> b({cpp_v2});\n")
                    f.write(f"        Bit<{w_res}> result = {cpp_expr};\n")
                else: # Unary
                    f.write(f"        Bit<{w_res}> result = {cpp_expr};\n")

                f.write(f'        std::cout << "Test: {line.strip()} -> Result: " << result.to_string() << std::endl;\n')
                f.write("    }\n")
            except (ValueError, KeyError) as e:
                print(f"Skipping malformed line: {line.strip()} -> {e}")

        f.write('    return 0;\n}\n')

def compare_results():
    """Performs an intelligent comparison of the result files."""
    print("--- Comparing Verilog (Golden) vs. C++ (DUT) ---")
    with open("verilog_results.txt", "r") as fv, open("cpp_results.txt", "r") as fc:
        verilog_lines = [line for line in fv if line.strip()]
        cpp_lines = [line for line in fc if line.strip()]

    errors = 0
    for i, (vline, cline) in enumerate(zip(verilog_lines, cpp_lines)):
        test_case = vline.split("->")[0]
        verilog_res = vline.split(":")[-1].strip()
        cpp_res = cline.split(":")[-1].strip()

        # Check for the special division-by-zero case
        is_div_by_zero = re.search(r"\s*(DIV|MOD),\s*\d+,\s*[^,]+,\s*\d+,\s*0\s*,", test_case)

        is_match = False
        if is_div_by_zero:
            # For div/mod by zero, Verilog should be all 'x's and C++ all 'f's.
            is_verilog_x = all(c == 'x' for c in verilog_res)
            is_cpp_f = all(c == 'f' for c in cpp_res)
            if is_verilog_x and is_cpp_f:
                is_match = True
        else:
            # For all other cases, they must match exactly.
            if verilog_res == cpp_res:
                is_match = True

        if not is_match:
            errors += 1
            print(f"❌ Mismatch on line {i+1}:")
            print(f"   Test: {test_case}")
            print(f"   Golden (Verilog): {verilog_res}")
            print(f"   Result (C++):     {cpp_res}")

    if errors == 0:
        print("✅ All tests passed!")
    else:
        print(f"\nFound {errors} mismatch(es).")
        sys.exit(1)

# Main execution logic
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_results()
    else:
        try:
            with open("test_vectors.txt", "r") as f:
                lines = f.readlines()
            generate_verilog(lines)
            generate_cpp(lines)
            print("✅ Successfully generated testbench.v and test_runner.cpp")
        except FileNotFoundError:
            print("❌ Error: test_vectors.txt not found. Please create it.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
