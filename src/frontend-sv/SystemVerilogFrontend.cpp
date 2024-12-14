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
#include "SurelogTranslator.hpp"
#include "uhdm/module_inst.h"
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

std::unique_ptr<cir::Ast>
SystemVerilogFrontend::compileSvToCir(std::vector<std::string> sources) {
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
        return nullptr;

    // Go to the next step
    return translateAst(vpi_design);
}

std::unique_ptr<cir::Ast>
SystemVerilogFrontend::translateAst(vpiHandle design_h) {
    auto iter = vpi_iterate(UHDM::uhdmtopModules, design_h);
    auto ast = std::make_unique<cir::Ast>();
    SurelogTranslator translator(*ast); 

    while (vpiHandle mod_h = vpi_scan(iter)) {
        auto mod_name = vpi_get_str(vpiName, mod_h);

        auto handle = reinterpret_cast<const uhdm_handle *>(mod_h);
        auto mod_handle = reinterpret_cast<const UHDM::module_inst *>(handle->object);

        spdlog::debug("Translating module {}", mod_name);

        auto mod_idx = translator.parseModule(*mod_handle);
        ast->setTopModule(mod_idx);
    }

    return ast;
}

SystemVerilogFrontend::~SystemVerilogFrontend() {
    if (m_compiler != nullptr)
        SURELOG::shutdown_compiler(m_compiler);
}

} // namespace cudalator
