#pragma once

#include <cstdint>
#include <vector>


template <typename S>
struct Process {
public:
    typedef void (*process_signature_t)(S *prev, S *next, size_t len);
    
    Process() = delete;
    Process(process_signature_t fp, std::vector<uint32_t> sens) : function_pointer(fp), sensitivity(sens) {}



    process_signature_t function_pointer;
    std::vector<uint32_t> sensitivity;
};
