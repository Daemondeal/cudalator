#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Surelog/API/Surelog.h>
#include <Surelog/CommandLine/CommandLineParser.h>
#include <Surelog/ErrorReporting/ErrorContainer.h>
#include <Surelog/SourceCompile/SymbolTable.h>

#include "cir/CIR.h"
#include "uhdm/vpi_user.h"

namespace cudalator {

class SystemVerilogFrontend {
public:
    SystemVerilogFrontend();
    ~SystemVerilogFrontend();

    std::unique_ptr<cir::Ast>
    compileSvToCir(std::vector<std::string> sources, bool print_uhdm_ast);

private:

    std::unique_ptr<cir::Ast> translateAst(vpiHandle design_h);

    SURELOG::scompiler *m_compiler;

    std::unique_ptr<SURELOG::SymbolTable> m_symbol_table;
    std::unique_ptr<SURELOG::ErrorContainer> m_errors;
    std::unique_ptr<SURELOG::CommandLineParser> m_clp;
};

} // namespace cudalator
