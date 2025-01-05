#include "SystemVerilogFrontend.hpp"

#include <cstring>
#include <filesystem>
#include <memory>

#include <Surelog/API/Surelog.h>
#include <Surelog/CommandLine/CommandLineParser.h>
#include <Surelog/ErrorReporting/ErrorContainer.h>
#include <Surelog/SourceCompile/SymbolTable.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <uhdm/ElaboratorListener.h>
#include <uhdm/VpiListener.h>
#include <uhdm/uhdm.h>
#include <uhdm/vpi_user.h>
#include <uhdm/vpi_visitor.h>

#include "FrontendError.hpp"
#include "SurelogTranslator.hpp"
#include "uhdm/module_inst.h"
#include "uhdm/uhdm_types.h"

namespace cudalator {

SystemVerilogFrontend::SystemVerilogFrontend() : m_compiler(nullptr) {
    m_symbol_table = std::make_unique<SURELOG::SymbolTable>();
    m_errors = std::make_unique<SURELOG::ErrorContainer>(m_symbol_table.get());
    m_clp = std::make_unique<SURELOG::CommandLineParser>(
        m_errors.get(), m_symbol_table.get(), false, false);
}

std::unique_ptr<cir::Ast>
SystemVerilogFrontend::compileSvToCir(std::vector<std::string> sources,
                                      bool print_udhm_ast) {
    // Set parameters
    m_clp->noPython();
    m_clp->setMuteStdout();
    m_clp->setwritePpOutput(true);
    m_clp->setParse(true);
    m_clp->setCompile(true);
    m_clp->setElaborate(true); // Request Surelog instance tree elaboration
    m_clp->setElabUhdm(true);  // Request UHDM Uniquification/Elaboration

    bool all_exist = true;
    for (auto source : sources) {
        if (!std::filesystem::exists(source)) {
            all_exist = false;
            spdlog::error("Cannot find file \"{}\"", source);
        }
    }

    if (!all_exist)
        return nullptr;

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

    if (print_udhm_ast) {
        spdlog::info("Printing udhm ast:");
        UHDM::visit_object(vpi_design, std::cout);
    }

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
        auto mod_handle =
            reinterpret_cast<const UHDM::module_inst *>(handle->object);

        spdlog::debug("Translating module {}", mod_name);

        auto mod_idx = translator.parseModule(*mod_handle);
        if (translator.getErrors().size() > 0) {
            auto path = vpi_get_str(vpiFile, mod_h);
            auto fpath = std::filesystem::path(path);
            auto file_name = fpath.filename().string();

            for (auto& error : translator.getErrors()) {
                reportError(error, file_name);
            }
            return nullptr;
        }
        ast->setTopModule(mod_idx);
    }

    return ast;
}

void SystemVerilogFrontend::reportError(FrontendError& error,
                                        std::string filename) {
    switch (error.kind()) {
    case FrontendErrorKind::Unsupported: {
        spdlog::error("{} (line {}) Unsupported: {}", filename, error.loc().line,
                      error.message());
    } break;
    case FrontendErrorKind::Todo: {
        spdlog::error("{} (line {}) TODO: {}", filename, error.loc().line,
                      error.message());
    } break;
    case FrontendErrorKind::Other:
    default: {
        spdlog::error("{} (line {}): {}", filename, error.loc().line, error.message());
    } break;
    }
}

SystemVerilogFrontend::~SystemVerilogFrontend() {
    if (m_compiler != nullptr)
        SURELOG::shutdown_compiler(m_compiler);
}

} // namespace cudalator
