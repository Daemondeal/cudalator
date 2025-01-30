#pragma once

#include "CppEmitter.hpp"
#include "cir/CIR.h"
#include <filesystem>
namespace cudalator {

class CodeGenerator {
public:
    CodeGenerator() = delete;
    CodeGenerator(const cir::Ast& ast, CppEmitter header, CppEmitter source);

    static void generateCode(const cir::Ast& ast, std::filesystem::path headerPath,
                         std::filesystem::path sourcePath);


private:
    enum class StatesArray {
        Prev,
        Next
    };

    void start();

    void generateSignalsStruct();
    void generateProcessDeclarations();

    void generateProcessCode(cir::ProcessIdx proc_idx);

    void generateStatement(const cir::Statement &stmt);
    void generateExpr(const cir::Expr &expr, StatesArray arr);

    void generateSignalDeclaration(CppEmitter& emitter, const cir::Signal& signal);
    void generateType(CppEmitter& emitter, const cir::Type& type);

    void generateProcessSignature(CppEmitter& emitter, cir::ProcessIdx proc_idx);

    std::string m_state_name;

    const cir::Ast& m_ast;
    CppEmitter m_header;
    CppEmitter m_source;
};

} // namespace cudalator
