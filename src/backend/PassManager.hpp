#pragma once

#include "cir/CIR.h"
namespace cudalator {

class PassManager {
public:
    PassManager();

    void runPasses(cir::Ast& ast);
};

} // namespace cudalator
