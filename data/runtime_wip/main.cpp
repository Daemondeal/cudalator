#include "runtime.hpp"
#include <iostream>

int main() {
    // Create Bit objects with different bit widths
    Bit<32> bit32_1;
    Bit<32> bit32_2;
    Bit<64> bit64_1;
    Bit<64> bit64_2;

    bit32_1 = 0x12345678;
    bit32_2 = 0x87654321;
    bit64_1 = 0x123456789ABCDEF0ULL;
    bit64_2 = 0x0FEDCBA987654321ULL;

    // Perform arithmetic operations
    Bit<32> bit32_sum = bit32_1 + bit32_2;
    Bit<32> bit32_diff = bit32_1 - bit32_2;
    Bit<32> bit32_prod = bit32_1 * bit32_2;

    Bit<64> bit64_sum = bit64_1 + bit64_2;
    Bit<64> bit64_diff = bit64_1 - bit64_2;
    Bit<64> bit64_prod = bit64_1 * bit64_2;

    // Perform bitwise operations
    Bit<32> bit32_and = bit32_1 & bit32_2;
    Bit<32> bit32_or = bit32_1 | bit32_2;
    Bit<32> bit32_xor = bit32_1 ^ bit32_2;
    Bit<32> bit32_not = ~bit32_1;

    Bit<64> bit64_and = bit64_1 & bit64_2;
    Bit<64> bit64_or = bit64_1 | bit64_2;
    Bit<64> bit64_xor = bit64_1 ^ bit64_2;
    Bit<64> bit64_not = ~bit64_1;

    // Perform shift operations
    Bit<32> bit32_shl = bit32_1 << 4;
    Bit<32> bit32_shr = bit32_1 >> 4;

    Bit<64> bit64_shl = bit64_1 << 8;
    Bit<64> bit64_shr = bit64_1 >> 8;

    // Print results
    std::cout << "bit32_1: " << bit32_1.to_string() << std::endl;
    std::cout << "bit32_2: " << bit32_2.to_string() << std::endl;
    std::cout << "bit32_sum: " << bit32_sum.to_string() << std::endl;
    std::cout << "bit32_diff: " << bit32_diff.to_string() << std::endl;
    std::cout << "bit32_prod: " << bit32_prod.to_string() << std::endl;
    std::cout << "bit32_and: " << bit32_and.to_string() << std::endl;
    std::cout << "bit32_or: " << bit32_or.to_string() << std::endl;
    std::cout << "bit32_xor: " << bit32_xor.to_string() << std::endl;
    std::cout << "bit32_not: " << bit32_not.to_string() << std::endl;
    std::cout << "bit32_shl: " << bit32_shl.to_string() << std::endl;
    std::cout << "bit32_shr: " << bit32_shr.to_string() << std::endl;

    std::cout << "bit64_1: " << bit64_1.to_string() << std::endl;
    std::cout << "bit64_2: " << bit64_2.to_string() << std::endl;
    std::cout << "bit64_sum: " << bit64_sum.to_string() << std::endl;
    std::cout << "bit64_diff: " << bit64_diff.to_string() << std::endl;
    std::cout << "bit64_prod: " << bit64_prod.to_string() << std::endl;
    std::cout << "bit64_and: " << bit64_and.to_string() << std::endl;
    std::cout << "bit64_or: " << bit64_or.to_string() << std::endl;
    std::cout << "bit64_xor: " << bit64_xor.to_string() << std::endl;
    std::cout << "bit64_not: " << bit64_not.to_string() << std::endl;
    std::cout << "bit64_shl: " << bit64_shl.to_string() << std::endl;
    std::cout << "bit64_shr: " << bit64_shr.to_string() << std::endl;

    return 0;
}
