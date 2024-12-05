#include "Entry.hpp"

#include <iostream>

#include <Surelog/API/Surelog.h>
#include <Surelog/CommandLine/CommandLineParser.h>
#include <Surelog/ErrorReporting/ErrorContainer.h>
#include <Surelog/SourceCompile/SymbolTable.h>

// UHDM
#include <uhdm/ElaboratorListener.h>
#include <uhdm/VpiListener.h>
#include <uhdm/uhdm.h>
#include <uhdm/vpi_user.h>

#include "SampleListener.hpp"


namespace cudalator {

static bool run_sample_listener(const vpiHandle &design_handle) {
    SampleListener listener;

    std::cout << "Start of design traversal\n";
    listener.listenDesigns({design_handle});
    std::cout << "End design traversal\n";

    return true;
}

void compile_sv_to_cil(std::vector<std::string> sources) {
    SURELOG::SymbolTable *const symbolTable = new SURELOG::SymbolTable();
    SURELOG::ErrorContainer *const errors =
        new SURELOG::ErrorContainer(symbolTable);
    SURELOG::CommandLineParser *const clp =
        new SURELOG::CommandLineParser(errors, symbolTable, false, false);


    // Set parameters
    clp->noPython();
    clp->setMuteStdout();
    clp->setwritePpOutput(true);
    clp->setParse(true);
    clp->setCompile(true);
    clp->setElaborate(true); // Request Surelog instance tree elaboration
    clp->setElabUhdm(true);  // Request UHDM Uniquification/Elaboration

    // NOTE(Pietro): A bit hacky but it's the easiest way I found to give it the input file
    std::cerr << "FIXME: ONLY COMPILING THE FIRST SOURCE\n";
    std::string path = sources[0];
    char const *args[2] = {"", path.c_str()};
    clp->parseCommandLine(2, args);

    errors->printMessages(clp->muteStdout());

    // Compile Design
    vpiHandle vpi_design = nullptr;
    SURELOG::scompiler *compiler = nullptr;

    compiler = SURELOG::start_compiler(clp);
    vpi_design = SURELOG::get_uhdm_design(compiler);

    // Handle errors
    auto stats = errors->getErrorStats();
    errors->printStats(stats, false);
    if (vpi_design == nullptr)
        return;
        // return false;

    // Go to the next step
    auto success = run_sample_listener(vpi_design);


    // Shutdown compiler
    SURELOG::shutdown_compiler(compiler);
    delete clp;
    delete symbolTable;
    delete errors;

    return;
    // return success;
}

}
