#pragma once

#include <fmt/format.h>
#include <fmt/os.h>
#include <vector>
#include <memory>

#include "Bit.hpp"
#include "../codegen/module.hpp"

class VcdDump {
public:
    VcdDump() = delete;
    VcdDump(std::unique_ptr<fmt::ostream> &&outfile, int idx)
        : m_outfile(std::move(outfile)), m_idx(idx) {}

    void print_header() {
        m_outfile->print("$version\n\tCudalator\n$end\n");
        m_outfile->print("$timescale 1ns $end\n");

        state_vcd_dump_names(*m_outfile);

        m_outfile->print("$enddefinitions $end\n");
    }

    void dump(std::vector<StateType> &state, int cycle) {
        m_outfile->print("#{}\n", cycle);
        state_vcd_dump_values(state.data(), m_idx, *m_outfile);
    }
private:

    std::unique_ptr<fmt::ostream> m_outfile;
    int m_idx;
};

template <int N>
static std::string vcd_dump_value(Bit<N> value) {
    return value.to_binary_string();
}

static std::string vcd_dump_value(uint64_t value) {
    return fmt::format("{:b}", value);
}
