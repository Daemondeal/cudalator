#include "adder_cpp.hpp"
#include <iostream>

void run_test(int val_a, int val_b) {
    Bit<8> a(val_a);
    Bit<8> b(val_b);

    auto result_pair = adder_cpp(a, b);
    Bit<8> c = result_pair.first;
    Bit<1> cout = result_pair.second;

    std::cout << "TEST a=" << val_a << ", b=" << val_b
              << " -> c=" << c.to_string() << ", cout=" << cout.to_string()
              << std::endl;
}

int main() {
    std::cout << "--- C++ Adder Test ---" << std::endl;

    // same test vectors as the Verilog testbench
    run_test(10, 5);
    run_test(255, 1);
    run_test(0x80, 0x80);

    return 0;
}
