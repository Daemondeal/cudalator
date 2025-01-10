#include "CodeGenerator.hpp"
#include "../utils.hpp"
#include "CppEmitter.hpp"
#include "cir/CIR.h"
#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <spdlog/spdlog.h>

namespace cudalator {

CodeGenerator::CodeGenerator(const cir::Ast& ast, CppEmitter header,
                             CppEmitter source)
    : m_ast(ast), m_header(header), m_source(source) {

    m_state_name = m_ast.getTopModule().name();
}

void CodeGenerator::generateCode(const cir::Ast& ast,
                                 std::filesystem::path headerPath,
                                 std::filesystem::path sourcePath) {

    std::ofstream sourceStream(sourcePath);
    std::ofstream headerStream(headerPath);

    CppEmitter source(sourceStream);
    CppEmitter header(headerStream);

    CodeGenerator generator(ast, header, source);

    generator.start();
}

void CodeGenerator::start() {
    spdlog::debug("Starting Codegen");

    m_header.emitNamespaceStart("cudalator_codegen");

    m_header.emitLine();
    m_source.emitLine();
    generateSignalsStruct();

    m_header.emitLine();
    generateProcessDeclarations();

    m_header.emitLine();
    m_header.emitNamespaceEnd("cudalator_codegen");

    m_source.emitNamespaceStart("cudalator_codegen");
    m_source.emitLine();

    auto& top = m_ast.getTopModule();
    for (auto proc_idx : top.processes()) {
        generateProcessCode(proc_idx);
        m_source.emitLine();
    }

    m_source.emitLine();
    m_source.emitNamespaceEnd("cudalator_codegen");
}

void CodeGenerator::generateProcessCode(cir::ProcessIdx proc_idx) {
    generateProcessSignature(m_source, proc_idx);
    m_source.emitBlockStart();

    m_source.emitIndent();
    m_source.emitLine("int tid = blockDim.x * blockIdx.x + threadIdx.x;");

    auto& proc = m_ast.getNode(proc_idx);
    auto& stmt = m_ast.getNode(proc.statement());

    // If the process contains a block, there is no need to add extra braces
    // to the resulting C++ statement.
    if (stmt.kind() == cir::StatementKind::Block) {
        for (auto& sub_idx : stmt.statements()) {
            generateStatement(m_ast.getNode(sub_idx));
        }
    } else {
        generateStatement(stmt);
    }

    m_source.emitBlockEnd();
}

void CodeGenerator::generateExpr(const cir::Expr& expr, StatesArray arr) {
    if (expr.isUnary()) {
        // TODO
    } else if (expr.isBinary()) {
        // TODO
    }

    switch (expr.kind()) {
    case cir::ExprKind::SignalRef: {
        auto& signal = m_ast.getNode(expr.signal());

        if (arr == StatesArray::Next) {
            m_source.emit("next");
        } else if (arr == StatesArray::Prev) {
            m_source.emit("prev");
        }

        m_source.emit("[tid].");


        m_source.emitName(signal.name());
    } break;

    case cir::ExprKind::Invalid:
    case cir::ExprKind::Constant:
    case cir::ExprKind::UnaryMinus:
    case cir::ExprKind::UnaryPlus:
    case cir::ExprKind::Not:
    case cir::ExprKind::BinaryNegation:
    case cir::ExprKind::ReductionAnd:
    case cir::ExprKind::ReductionNand:
    case cir::ExprKind::ReductionOr:
    case cir::ExprKind::ReductionNor:
    case cir::ExprKind::ReductionXor:
    case cir::ExprKind::ReductionXnor:
    case cir::ExprKind::Posedge:
    case cir::ExprKind::Negedge:
    case cir::ExprKind::Subtraction:
    case cir::ExprKind::Division:
    case cir::ExprKind::Modulo:
    case cir::ExprKind::Equality:
    case cir::ExprKind::NotEquality:
    case cir::ExprKind::GreaterThan:
    case cir::ExprKind::GreaterThanEq:
    case cir::ExprKind::LessThan:
    case cir::ExprKind::LessThanEq:
    case cir::ExprKind::LeftShift:
    case cir::ExprKind::RightShift:
    case cir::ExprKind::Addition:
    case cir::ExprKind::Multiplication:
    case cir::ExprKind::LogicalAnd:
    case cir::ExprKind::LogicalOr:
    case cir::ExprKind::BitwiseAnd:
    case cir::ExprKind::BitwiseOr:
    case cir::ExprKind::BitwiseXor:
    case cir::ExprKind::BitwiseXnor:
    case cir::ExprKind::Concatenation:
    case cir::ExprKind::PartSelect:
    case cir::ExprKind::BitSelect:
        CD_ASSERT_MSG(false, "codegen not implemented for expr");
        break;
    }
}

void CodeGenerator::generateStatement(const cir::Statement& stmt) {
    switch (stmt.kind()) {

    case cir::StatementKind::NonBlockingAssignment: {
        auto& lhs = m_ast.getNode(stmt.lhs());
        auto& rhs = m_ast.getNode(stmt.rhs());

        m_source.emitIndent();
        generateExpr(lhs, StatesArray::Next);
        m_source.emit(" = ");
        generateExpr(rhs, StatesArray::Prev);
        m_source.emitLine(";");

        m_source.emitIndent();
        generateExpr(lhs, StatesArray::Prev);
        m_source.emit(" = ");
        generateExpr(lhs, StatesArray::Next);
        m_source.emitLine(";");
    } break;

    case cir::StatementKind::Assignment: {
        auto& lhs = m_ast.getNode(stmt.lhs());
        auto& rhs = m_ast.getNode(stmt.rhs());

        m_source.emitIndent();
        generateExpr(lhs, StatesArray::Next);
        m_source.emit(" = ");
        generateExpr(rhs, StatesArray::Prev);
        m_source.emitLine(";");
    } break;

    case cir::StatementKind::Block:
    case cir::StatementKind::If:
    case cir::StatementKind::IfElse:
    case cir::StatementKind::While:
    case cir::StatementKind::DoWhile:
    case cir::StatementKind::Repeat:
    case cir::StatementKind::For:
    case cir::StatementKind::Case:
    case cir::StatementKind::Foreach:
    case cir::StatementKind::Forever:
    case cir::StatementKind::Return:
    case cir::StatementKind::Break:
    case cir::StatementKind::Continue:
    case cir::StatementKind::Null:
    case cir::StatementKind::Invalid: {

    } break;
    }
}

void CodeGenerator::generateProcessDeclarations() {
    auto& top = m_ast.getTopModule();

    for (auto proc_idx : top.processes()) {
        auto& proc = m_ast.getNode(proc_idx);

        generateProcessSignature(m_header, proc_idx);
        m_header.emitLine(";");
    }
}

void CodeGenerator::generateProcessSignature(CppEmitter& emitter,
                                             cir::ProcessIdx proc_idx) {
    int32_t idx = proc_idx.idx;

    emitter.emitIndent();
    emitter.emit("__global__ void ");
    emitter.emit(fmt::format("__process_{}", idx));
    emitter.emit(" (");
    emitter.emitName(m_state_name);
    emitter.emit(" *prev, ");
    emitter.emitName(m_state_name);
    emitter.emit(" *next, size_t len)");
}

void CodeGenerator::generateSignalsStruct() {
    auto& top = m_ast.getTopModule();
    auto& scope = m_ast.getNode(top.scope());

    m_header.emitStructStart(top.name());

    for (auto signal_idx : scope.signals()) {
        auto& signal = m_ast.getNode(signal_idx);
        generateSignalDeclaration(m_header, signal);
    }

    m_header.emitStructEnd(top.name());
}

void CodeGenerator::generateSignalDeclaration(CppEmitter& emitter,
                                              const cir::Signal& signal) {
    emitter.emitIndent();
    generateType(emitter, m_ast.getNode(signal.type()));
    emitter.emit(" ");
    emitter.emitName(signal.name());
    emitter.emitLine(";");
}

// TODO: This will have to be modified to fit the runtime
void CodeGenerator::generateType(CppEmitter& emitter, const cir::Type& type) {
    switch (type.kind()) {
    case cir::TypeKind::Invalid: {
        CD_UNREACHABLE("Invalid Type");
    } break;
    case cir::TypeKind::Bit: {
        emitter.emit(fmt::format("sv_bit<{}>", type.size()));
    } break;
    case cir::TypeKind::Logic: {
        emitter.emit(fmt::format("sv_logic<{}>", type.size()));
    } break;
    case cir::TypeKind::Integer: {
        emitter.emit("sv_integer");
    } break;

    case cir::TypeKind::Int: {
        emitter.emit("sv_int");
    } break;
    }
}

} // namespace cudalator
