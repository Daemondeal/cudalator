#pragma once

#include <memory>
#include <vector>
#include <string>

#include <Surelog/API/Surelog.h>
#include <Surelog/ErrorReporting/ErrorContainer.h>
#include <Surelog/CommandLine/CommandLineParser.h>
#include <Surelog/SourceCompile/SymbolTable.h>

namespace cudalator {

class SystemVerilogFrontend {
private:
    SURELOG::scompiler *m_compiler;

    std::unique_ptr<SURELOG::SymbolTable> m_symbol_table;
    std::unique_ptr<SURELOG::ErrorContainer> m_errors;
    std::unique_ptr<SURELOG::CommandLineParser> m_clp;

public:
    SystemVerilogFrontend();
    ~SystemVerilogFrontend();

    void compile_sv_to_cil(std::vector<std::string> sources);
};

}
