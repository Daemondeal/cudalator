#pragma once

#include "ChangeType.hpp"
#include <cstdint>
#include <string>
#include <vector>

template <typename S>
struct Process {
public:
    typedef void (*process_signature_t)(S* state, size_t len);

    Process() = delete;
    Process(
        std::string name,
        process_signature_t fp,
        std::vector<std::tuple<uint32_t, ChangeType>> sens
    )
        : name(name)
        , function_pointer(fp)
        , sensitivity(sens) {
    }

    std::string name;
    process_signature_t function_pointer;
    std::vector<std::tuple<uint32_t, ChangeType>> sensitivity;
};
