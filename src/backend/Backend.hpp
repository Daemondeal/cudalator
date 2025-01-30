#pragma once

#include "cir/CIR.h"
#include <memory>

namespace cudalator {

void run_backend(std::unique_ptr<cir::Ast> ast, bool print_ast);

}
