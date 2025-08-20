#include "runtime.hpp" // Make sure this path is correct
#include <iostream>

// Helper function to print test results
void check(const std::string& test_name, const Bit<1>& result, bool expected) {
    bool passed = (static_cast<bool>(result) == expected);
    std::cout << test_name << ": " << (passed ? "PASS" : "FAIL")
              << " (Got: " << result.to_string()
              << ", Expected: " << (expected ? "1" : "0") << ")" << std::endl;
}

int main() {
    std::cout << "--- Testing Bit::inside() function ---" << std::endl;

    // Test 1: Value is inside the set
    // INSIDE, 8, 50, 10;50;100 -> should be true
    {
        Bit<8> val{50};
        check("Test 1 (50 in {10,50,100})",
              Bit<8>::inside(val, Bit<8>{10}, Bit<8>{50}, Bit<8>{100}), true);
    }

    // Test 2: Value is not inside the set
    // INSIDE, 8, 99, 10;50;100 -> should be false
    {
        Bit<8> val{99};
        check("Test 2 (99 in {10,50,100})",
              Bit<8>::inside(val, Bit<8>{10}, Bit<8>{50}, Bit<8>{100}), false);
    }

    // Test 3: Test with a single-element set
    // INSIDE, 16, 1234, 1234 -> should be true
    {
        Bit<16> val{1234};
        check("Test 3 (1234 in {1234})", Bit<16>::inside(val, Bit<16>{1234}),
              true);
    }

    // Test 4: Test with zero
    // INSIDE, 32, 0, 1;2;3;0;5 -> should be true
    {
        Bit<32> val{0};
        check("Test 4 (0 in {1,2,3,0,5})",
              Bit<32>::inside(val, Bit<32>{1}, Bit<32>{2}, Bit<32>{3},
                              Bit<32>{0}, Bit<32>{5}),
              true);
    }

    // Test 5: Test with different bit-widths in the set
    {
        Bit<16> val{42};
        check("Test 5 (42 in {Bit<8>, Bit<16>})",
              Bit<16>::inside(val, Bit<8>{10}, Bit<16>{42}), true);
    }

    return 0;
}
