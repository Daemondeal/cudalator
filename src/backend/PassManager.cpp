#include "PassManager.hpp"
#include "passes/PopulateSensitivityList.hpp"

namespace cudalator {

PassManager::PassManager() {}

void PassManager::runPasses(cir::Ast& ast) {
    PopulateSensitivityList pass_psm(ast);

    pass_psm.runPass();
}

} // namespace cudalator
