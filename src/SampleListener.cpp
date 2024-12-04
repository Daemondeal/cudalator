#include "include/SampleListener.h"

#include <uhdm/uhdm.h>

#include <iostream>

void SampleListener::enterModule_inst(const UHDM::module_inst *object,
                                      vpiHandle handle) {
    std::string_view instName = object->VpiName();
    m_flatTraversal =
        (instName.empty()) && ((object->VpiParent() == nullptr) ||
                               ((object->VpiParent() != nullptr) &&
                                (object->VpiParent()->VpiType() != vpiModule)));

    if (m_flatTraversal) {
        std::cout << "Module Definition: " << object->VpiDefName() << std::endl;
        return;
    }

    std::cout << "Module Instance: " << object->VpiFullName() << std::endl;
}

void SampleListener::enterCont_assign(const UHDM::cont_assign *object,
                                      vpiHandle handle) {
    if (m_flatTraversal)
        return;

    std::cout << "  enterCont_assign " << intptr_t(object) << " "
              << object->UhdmId() << std::endl;

    auto lhs = object->Lhs();
    auto rhs = object->Rhs();
    std::cout << "Continuous assignment found!\n";
    std::cout << "assigning " << rhs->VpiName() << " to " << lhs->VpiName()
              << "\n";
}
