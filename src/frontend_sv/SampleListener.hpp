#pragma once

#include "uhdm/VpiListener.h"

class SampleListener final : public UHDM::VpiListener {
    void enterModule_inst(const UHDM::module_inst *object,
                          vpiHandle handle) final;
    void enterCont_assign(const UHDM::cont_assign *object,
                          vpiHandle handle) final;

private:
    bool m_flatTraversal;
};
