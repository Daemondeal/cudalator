#pragma once


#include "cir/CIR.h"
#include "uhdm/vpi_user.h"
#include <cstdint>
#include <string>

std::string getVpiTypeName(uint32_t type);

cir::ExprKind vpiUnaryOp(uint32_t op_t) {
    switch (op_t) {
    case vpiPosedgeOp:
        return cir::ExprKind::Posedge;
    case vpiNegedgeOp:
        return cir::ExprKind::Negedge;
    default:
        return cir::ExprKind::Invalid;

    }
}
