#include "runtime.hpp"
#include <iostream>

int main() {
    Bit<8> a = {10};
    Bit<8> b = {10};

    std::cout << "Initial a: " << a.to_string() << std::endl; // a
    std::cout << "Initial b: " << b.to_string() << std::endl; // a

    // Using the PREFIX operator calls operator++()
    Bit<8> res_prefix = ++a;
    std::cout << "Result of ++a: " << res_prefix.to_string() << std::endl; // b
    std::cout << "Final a: " << a.to_string() << std::endl;                // b

    std::cout << "---" << std::endl;

    // Using the POSTFIX operator calls operator++(int)
    Bit<8> res_postfix = b++;
    std::cout << "Result of b++: " << res_postfix.to_string() << std::endl; // a
    std::cout << "Final b: " << b.to_string() << std::endl;                 // b

    return 0;
}
