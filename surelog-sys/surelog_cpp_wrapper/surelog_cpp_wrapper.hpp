#pragma once

// TODO: Find a MacOS-Compatible way of including size_t
// #include <cstddef>
#include <uhdm/vpi_user.h>
#include <uhdm/sv_vpi_user.h>
#include <uhdm/uhdm_vpi_user.h>

struct SystemVerilogDesign;

extern "C" {

SystemVerilogDesign* design_create();
vpiHandle design_compile(SystemVerilogDesign* d, char const* const* sources, unsigned long long sources_len, char const* top_module);
void design_free(SystemVerilogDesign* d);

vpiHandle design_top_entity(vpiHandle design);

void vpi_visit(vpiHandle obj_handle);

}
