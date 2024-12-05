#include "frontend-sv/Entry.hpp"

#include <iostream>

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "USAGE: cudalator-compiler <path-to-sv-file>\n";
        return -1;
    }
    std::string source_path(argv[1]);

    cudalator::compile_sv_to_cil({source_path});

    return 0;
}
