#include <Bit.hpp>

int main() {
    Bit<32> a;
    Bit<32> b;
    Bit<32> c;

    a = 1;
    b = 2;

    c = a + b;
    printf("%d\n", c.value);
}
