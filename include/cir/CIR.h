#pragma once

#include "GenericAst.h"
#include <string>
#include <string_view>
#include <variant>
namespace cir {

struct Signal;
struct Process;
struct Module;
struct Type;
struct Statement;
struct Expr;

// SystemVerilog types
enum class TypeKind {
    Wire,
    Reg,
    Integer,
    Logic,
};


struct Type {
public:
    TypeKind kind;
    uint32_t size;

    Type(TypeKind kind, uint32_t size) : kind(kind), size(size) {}
};

struct Signal {
public:
    std::string_view name;

    Type type;

    Signal(std::string_view name, Type type) : name(name), type(type) {}
};

enum class ExprKind {
    Binary,
    Unary,
    SignalRef,
    Number,
};

struct Expr {
public:
    ExprKind kind;

    Expr *lhs;
    Expr *rhs;

    Expr(ExprKind kind, Expr *lhs, Expr *rhs) : kind(kind), lhs(lhs), rhs(rhs) {}

};

struct Process {
    std::string_view name;

    std::vector<Statement *> statements;

    Process(std::string_view name) : name(name) {}
};


struct Module {
public:
    std::string_view name;

    std::vector<Process *> processes;

    Module(std::string_view name) : name(name), processes() {}

    void addProcess(Process *proc) {
        processes.push_back(proc);
    }
};

struct Ast : GenericAst<Signal, Process, Module> {
    using GenericAst::GenericAst;
};

} // namespace cir
