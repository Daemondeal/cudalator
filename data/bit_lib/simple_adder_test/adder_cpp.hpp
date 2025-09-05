#ifndef ADDER_CPP_HPP
#define ADDER_CPP_HPP

#include "../runtime_wip/runtime.hpp"
#include <utility> // For std::pair

std::pair<Bit<8>, Bit<1>> adder_cpp(const Bit<8>& a, const Bit<8>& b) {
    auto full_sum = a + b;
    // truncation
    Bit<8> c = full_sum;
    // estrazione carry out
    Bit<1> cout = full_sum >> 8;
    return {c, cout};
}

#endif // ADDER_CPP_HPP
