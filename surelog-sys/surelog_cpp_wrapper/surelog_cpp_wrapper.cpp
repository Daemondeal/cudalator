#include "surelog_cpp_wrapper.hpp"

#include <cstddef>
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

    vpiHandle compile_to_handle(char const* const* sources, size_t sources_len)
    {
        m_clp->noPython();
        m_clp->setMuteStdout();
        m_clp->setwritePpOutput(true);
        m_clp->setParse(true);
        m_clp->setCompile(true);
        m_clp->setElaborate(true); // Request Surelog instance tree elaboration
        m_clp->setElabUhdm(true); // Request UHDM Uniquification/Elaboration

        (void)sources_len;
        // TODO: Do this properly
        char const* args[] = { "", sources[0] };
        m_clp->parseCommandLine(2, args);

        m_compiler = SURELOG::start_compiler(m_clp.get());

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

vpiHandle design_compile(SystemVerilogDesign *d, char const* const* sources, unsigned long long sources_len) {
    return d->compile_to_handle(sources, sources_len);
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
