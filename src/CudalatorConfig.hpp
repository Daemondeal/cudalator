#pragma once

#include <vector>
#include <string>

struct CudalatorConfig {
public:
    std::vector<std::string> sources;

    bool verbose;

    bool print_udhm_ast;
private:
};
