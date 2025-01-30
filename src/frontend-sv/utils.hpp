#pragma once


#include "cir/CIR.h"
#include "uhdm/vpi_user.h"
#include <cstdint>
#include <string>

std::string getVpiTypeName(uint32_t type);

cir::ExprKind vpiUnaryOp(uint32_t op_t);
cir::ExprKind vpiBinaryOp(uint32_t op_t);

