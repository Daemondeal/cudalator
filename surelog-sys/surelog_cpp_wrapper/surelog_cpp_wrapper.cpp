#include "surelog_cpp_wrapper.hpp"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <uhdm/vpi_visitor.h>

#include <uhdm/uhdm_types.h>
#include <Surelog/API/Surelog.h>
#include <Surelog/CommandLine/CommandLineParser.h>
#include <Surelog/ErrorReporting/ErrorContainer.h>
#include <Surelog/SourceCompile/SymbolTable.h>

// Struct Implementation
struct SystemVerilogDesign {

public:
    SystemVerilogDesign()
        : m_compiler(nullptr)
    {
        m_symbol_table = std::make_unique<SURELOG::SymbolTable>();
        m_errors = std::make_unique<SURELOG::ErrorContainer>(m_symbol_table.get());
        m_clp = std::make_unique<SURELOG::CommandLineParser>(
            m_errors.get(), m_symbol_table.get(), false, false);
    }

    ~SystemVerilogDesign()
    {
        if (m_compiler != nullptr) {
            SURELOG::shutdown_compiler(m_compiler);
        }
    }

    vpiHandle compile_to_handle(char const* const* sources, size_t sources_len, char const* top_module)
    {
        m_clp->noPython();
        m_clp->setMuteStdout();
        m_clp->setwritePpOutput(true);
        m_clp->setParse(true);
        m_clp->setCompile(true);
        m_clp->setElaborate(true); // Request Surelog instance tree elaboration
        m_clp->setElabUhdm(true); // Request UHDM Uniquification/Elaboration

        if (strcmp(top_module, "") != 0) {
            m_clp->setTopLevelModule(top_module);
        }

        // The easiest way of passing mulitple files to Surelog is to give them to it via command line. 
        // This is a bit hacky, but it works fine.
        char const **args = new char const*[sources_len + 1];
        args[0] = "";
        for (size_t i = 0; i < sources_len; i++) {
            args[i+1] = sources[i];
        }

        m_clp->parseCommandLine(sources_len+1, args);


        m_compiler = SURELOG::start_compiler(m_clp.get());

        delete[] args;

        return SURELOG::get_uhdm_design(m_compiler);
    }

private:
    SURELOG::scompiler* m_compiler;

    std::unique_ptr<SURELOG::SymbolTable> m_symbol_table;
    std::unique_ptr<SURELOG::ErrorContainer> m_errors;
    std::unique_ptr<SURELOG::CommandLineParser> m_clp;
};

// Actual implementation of Rust API

SystemVerilogDesign *design_create() {
    auto *d =  new SystemVerilogDesign();
    return d;
}

vpiHandle design_compile(SystemVerilogDesign *d, char const* const* sources, unsigned long long sources_len, char const* top_module) {
    return d->compile_to_handle(sources, sources_len, top_module);
};

void design_free(SystemVerilogDesign *d) {
    delete d;
}

vpiHandle design_top_entity(vpiHandle design) {
    auto iter = vpi_iterate(UHDM::uhdmtopModules, design);
    while (vpiHandle mod_h = vpi_scan(iter)) {
        return mod_h;
    }

    return nullptr;
}

void vpi_visit(vpiHandle obj_handle) {
    UHDM::visit_object(obj_handle, std::cout);
}
