#pragma once

#include <vector>
#include <string>

struct CudalatorConfig {
public:
    std::vector<std::string> sources;

    std::string top_entity_name;

    bool verbose;

    bool print_udhm_ast;
private:
};
