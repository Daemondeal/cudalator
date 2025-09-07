#pragma once

#include "ChangeType.hpp"
#include <cstdint>
#include <string>
#include <vector>

template <typename S>
struct Process {
public:
    Process() = delete;
    Process(std::string name, size_t id,
            std::vector<std::tuple<uint32_t, ChangeType>> sens)
        : name(name), id(id), sensitivity(sens) {}

    std::string name;
    size_t id;
    std::vector<std::tuple<uint32_t, ChangeType>> sensitivity;
};
