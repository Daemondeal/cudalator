// tb_s5378.cpp
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include "verilated.h"
#include "Vs5378.h"

using namespace std;

class Random {
public:
    Random() = delete;
    Random(uint32_t seed) : m_state(seed) {}

    uint32_t get() {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return m_state;
    }

private:
    uint32_t m_state;
};

void tick(Vs5378* top) {
    top->CK = 1;
    top->eval();

    top->CK = 0;
    top->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    int instances = (argc > 1) ? atoi(argv[1]) : 256;

    Vs5378* top = new Vs5378;

    // List all inputs (CK is index 0, but we drive it in tick)
    vluint8_t* inputs[] = {
        &top->CK,
        &top->n3065gat,&top->n3066gat,&top->n3067gat,&top->n3068gat,&top->n3069gat,&top->n3070gat,
        &top->n3071gat,&top->n3072gat,&top->n3073gat,&top->n3074gat,&top->n3075gat,&top->n3076gat,
        &top->n3077gat,&top->n3078gat,&top->n3079gat,&top->n3080gat,&top->n3081gat,&top->n3082gat,
        &top->n3083gat,&top->n3084gat,&top->n3085gat,&top->n3086gat,&top->n3087gat,&top->n3088gat,
        &top->n3089gat,&top->n3090gat,&top->n3091gat,&top->n3092gat,&top->n3093gat,&top->n3094gat,
        &top->n3095gat,&top->n3097gat,&top->n3098gat,&top->n3099gat,&top->n3100gat
    };
    const int NUM_INPUTS = sizeof(inputs)/sizeof(inputs[0]);

    // Zero init
    for (int i = 0; i < NUM_INPUTS; ++i) *inputs[i] = 0;
    top->eval();

    Random rng(time(NULL));

    cout << "Number of instances: " << instances << endl;
    for (int cycle = 0; cycle < 1000; ++cycle) {
        for (int j = 0; j < instances; j++) {
            for (int i = 1; i < NUM_INPUTS; ++i) {
                *inputs[i] = rng.get() & 1;
            }

            tick(top);
            if (Verilated::gotFinish()) break;
        }
    }

    top->final();
    delete top;
    return 0;
}
