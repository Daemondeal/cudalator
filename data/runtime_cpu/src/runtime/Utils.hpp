#pragma once

#include <fmt/base.h>
#include <cstdint>

namespace cu {

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

static void progress_display(uint32_t current, uint32_t max) {
    constexpr uint32_t BAR_LENGTH = 20;

    uint32_t steps = (BAR_LENGTH * (current+1)) / max;

    fmt::print("\r[");
    for (int j = 0; j < steps; j++) {
        fmt::print("#");
    }

    for (int j = steps; j < BAR_LENGTH; j++) {
        fmt::print(".");
    }

    fmt::print("] {}/{}", current+1, max);
    fflush(stdout);
}

};
