#include "SystemVerilogFrontend.hpp"

#include <cstring>
#include <memory>

#include <Surelog/API/Surelog.h>
#include <Surelog/CommandLine/CommandLineParser.h>
#include <Surelog/ErrorReporting/ErrorContainer.h>
#include <Surelog/SourceCompile/SymbolTable.h>
#include <spdlog/spdlog.h>
#include <uhdm/ElaboratorListener.h>
#include <uhdm/VpiListener.h>
#include <uhdm/uhdm.h>
#include <uhdm/vpi_user.h>
#include <uhdm/vpi_visitor.h>

#include "SurelogParser.hpp"
#include "uhdm/uhdm_types.h"

namespace cudalator {

static bool run_sample_listener(const vpiHandle& design_handle) {
    SurelogParser parser;

    // visit and print to stdout
    vpiHandle top_entity = vpi_handle(vpiTopModule, design_handle);

    auto iterator = vpi_iterate(UHDM::uhdmtopModules, design_handle);

    while (vpiHandle mod_h = vpi_scan(iterator)) {
        auto str = vpi_get_str(vpiName, mod_h);

        if (strcmp("work@simple", str) == 0)
            UHDM::visit_object(mod_h, std::cout);
    }

    spdlog::debug("Starting the parser");

    parser.parse(design_handle);

    spdlog::debug("Parser Done");
    return true;
}

SystemVerilogFrontend::SystemVerilogFrontend() : m_compiler(nullptr) {
    m_symbol_table = std::make_unique<SURELOG::SymbolTable>();
    m_errors = std::make_unique<SURELOG::ErrorContainer>(m_symbol_table.get());
    m_clp = std::make_unique<SURELOG::CommandLineParser>(
        m_errors.get(), m_symbol_table.get(), false, false);
}

void SystemVerilogFrontend::compile_sv_to_cil(
    std::vector<std::string> sources) {
    // Set parameters
    m_clp->noPython();
    m_clp->setMuteStdout();
    m_clp->setwritePpOutput(true);
    m_clp->setParse(true);
    m_clp->setCompile(true);
    m_clp->setElaborate(true); // Request Surelog instance tree elaboration
    m_clp->setElabUhdm(true);  // Request UHDM Uniquification/Elaboration

    // NOTE(Pietro): A bit hacky but it's the easiest way I found
    //               to give it the input file
    spdlog::warn("FIXME: ONLY COMPILING THE FIRST SOURCE");
    std::string path = sources[0];
    char const *args[2] = {"", path.c_str()};
    m_clp->parseCommandLine(2, args);

    // Compile Design
    vpiHandle vpi_design = nullptr;
    m_compiler = SURELOG::start_compiler(m_clp.get());
    vpi_design = SURELOG::get_uhdm_design(m_compiler);

    if (vpi_design == nullptr)
        return;

    // Go to the next step
    auto success = run_sample_listener(vpi_design);

    return;
}

SystemVerilogFrontend::~SystemVerilogFrontend() {
    if (m_compiler != nullptr)
        SURELOG::shutdown_compiler(m_compiler);
}

} // namespace cudalator
