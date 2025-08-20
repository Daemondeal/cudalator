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

    Bit<4> condition{0b0100}; // non-zero, 4-bit condition
    Bit<8> val_if_true{100};
    Bit<8> val_if_false{200};
    Bit<8> result = Bit<8>::conditional(condition, val_if_true, val_if_false);

    std::cout << "Result: " << result.to_string()
              << std::endl; // in teoria 64 = 0x100
    return 0;
}
