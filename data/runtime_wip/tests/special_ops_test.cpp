#include "../runtime.hpp"
#include <iostream>
#include <string>

// formatter
void print_result(const std::string& test_id, const std::string& hex_val) {
    std::cout << "TEST " << test_id << " Result: " << hex_val << std::endl;
}

int main() {
    std::cout << "--- C++ Special Operations Test ---" << std::endl;

    // concat and replication tests
    print_result("CONCAT_1",
                 Bit<16>::concat(Bit<8>("0xAA"), Bit<8>("0xBB")).to_string());
    print_result("CONCAT_2",
                 Bit<20>::concat(Bit<8>("0xDE"), Bit<4>("0xA"), Bit<8>("0xDF"))
                     .to_string());
    // concatenating to the max width
    print_result("CONCAT_3_MAX_WIDTH",
                 Bit<128>::concat(Bit<64>("0xAAAAAAAAAAAAAAAA"),
                                  Bit<64>("0x5555555555555555"))
                     .to_string());

    print_result("REPLICATE_1",
                 Bit<16>::replicate<2>(Bit<8>("0xAB")).to_string());
    print_result("REPLICATE_2",
                 Bit<40>::replicate<5>(Bit<8>("0xF0")).to_string());
    // replicating a single bit
    print_result("REPLICATE_3_SINGLE_BIT",
                 Bit<8>::replicate<8>(Bit<1>("0x1")).to_string());

    // inside test
    print_result("INSIDE_1",
                 Bit<8>::inside(Bit<8>{50}, Bit<8>{10}, Bit<8>{50}, Bit<8>{100})
                     .to_string());
    print_result("INSIDE_2",
                 Bit<8>::inside(Bit<8>{99}, Bit<8>{10}, Bit<8>{50}, Bit<8>{100})
                     .to_string());
    print_result("INSIDE_3",
                 Bit<16>::inside(Bit<16>{1234}, Bit<16>{1234}).to_string());
    print_result("INSIDE_4",
                 Bit<32>::inside(Bit<32>{0}, Bit<32>{1}, Bit<32>{2}, Bit<32>{0})
                     .to_string());
    // value not in a set of different-width numbers
    print_result(
        "INSIDE_5_MIXED_WIDTH_FAIL",
        Bit<16>::inside(Bit<16>{999}, Bit<8>{10}, Bit<32>{998}).to_string());

    // increment/decrement
    Bit<8> a = {10};
    Bit<8> b = {10};
    print_result("INC_A_INITIAL", a.to_string());
    print_result("INC_B_INITIAL", b.to_string());
    Bit<8> res_prefix = ++a;
    print_result("INC_A_PREFIX_RES", res_prefix.to_string());
    print_result("INC_A_FINAL", a.to_string());
    Bit<8> res_postfix = b++;
    print_result("INC_B_POSTFIX_RES", res_postfix.to_string());
    print_result("INC_B_FINAL", b.to_string());
    // increment causing a overflow
    Bit<8> c = {255};
    print_result("INC_C_WRAP_INITIAL", c.to_string());
    c++;
    print_result("INC_C_WRAP_FINAL", c.to_string());

    // cond
    Bit<4> condition_true{0b0100};
    Bit<8> val_if_true{100};
    Bit<8> val_if_false{200};
    print_result("CONDITIONAL_1_TRUE",
                 Bit<8>::conditional(condition_true, val_if_true, val_if_false)
                     .to_string());
    // condition false (zero)
    Bit<4> condition_false{0};
    print_result("CONDITIONAL_2_FALSE",
                 Bit<8>::conditional(condition_false, val_if_true, val_if_false)
                     .to_string());

    return 0;
}
