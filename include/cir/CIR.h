#pragma once

#include "GenericAst.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cir {

struct Signal;
struct Process;
struct Module;
struct Type;
struct Statement;
struct Expr;

using SignalIdx = NodeIndex<Signal>;
using ProcessIdx = NodeIndex<Process>;
using ModuleIdx = NodeIndex<Module>;
using TypeIdx = NodeIndex<Type>;
using StatementIdx = NodeIndex<Statement>;
using ExprIdx = NodeIndex<Expr>;

class Loc {
public:
    uint32_t line;
    uint32_t column;
    Loc(uint32_t line, uint32_t col) : line(line), column(col) {}
};

struct NodeBase {
public:
    NodeBase(std::string_view name, Loc loc) : m_name(name), m_loc(loc) {}
    NodeBase() = delete;

    const std::string_view& name() const {
        return m_name;
    }

    Loc loc() const {
        return m_loc;
    }

private:
    std::string_view m_name;

    Loc m_loc;
};

struct Range {
public:
    explicit Range(uint32_t left, uint32_t right)
        : m_left(left), m_right(right) {}

    uint32_t left() {
        return m_left;
    }
    uint32_t right() {
        return m_right;
    }

private:
    uint32_t m_left;
    uint32_t m_right;
};

enum class TypeKind {
    Bit,
    Logic,
    Integer,
};

struct Type {
public:
    Type(TypeKind kind) : m_kind(kind) {}

    Type(TypeKind kind, TypeIdx subtype) : m_kind(kind), m_subtype(subtype) {}

    void addRange(Range range) {
        m_ranges.push_back(range);
    }

    TypeKind kind() const {
        return m_kind;
    }

    Range firstRange() const {
        return m_ranges[0];
    }

    const std::vector<Range>& ranges() const {
        return m_ranges;
    }

    TypeIdx subtype() const {
        return m_subtype;
    }

private:
    TypeKind m_kind;
    std::vector<Range> m_ranges;
    TypeIdx m_subtype;
};

enum class SignalDirection {
    Internal,
    Input,
    Output,
    Inout,
};

struct Signal : NodeBase {
public:
    explicit Signal(std::string_view name, Loc loc, TypeIdx type,
                    SignalDirection kind)
        : NodeBase(name, loc), m_type(type), m_kind(kind) {}

    TypeIdx type() const {
        return m_type;
    }

    SignalDirection kind() const {
        return m_kind;
    }

private:
    TypeIdx m_type;

    SignalDirection m_kind;
};

enum class ExprKind {
    Binary,
    Unary,
    SignalRef,
    Number,
};

struct Expr : NodeBase {
public:
    Expr(std::string_view name, Loc loc, ExprKind kind, ExprIdx lhs,
         ExprIdx rhs)
        : NodeBase(name, loc), m_kind(kind), m_lhs(lhs), m_rhs(rhs) {}

    Expr(std::string_view name, Loc loc, ExprKind kind, SignalIdx signal)
        : NodeBase(name, loc), m_kind(kind), m_signal(signal) {}

    Expr(std::string_view name, Loc loc, ExprKind kind, uint32_t constant)
        : NodeBase(name, loc), m_kind(kind), m_constant(constant) {}

    ExprIdx lhs() const {
        return m_lhs;
    }

    ExprIdx rhs() const {
        return m_rhs;
    }

    SignalIdx signal() const {
        return m_signal;
    }

    uint32_t constant() const {
        return m_constant;
    }

private:
    ExprKind m_kind;

    ExprIdx m_lhs;
    ExprIdx m_rhs;

    SignalIdx m_signal;
    uint32_t m_constant;
};

struct Process : NodeBase {
public:
    Process(std::string_view name, Loc loc) : NodeBase(name, loc) {}

    const std::vector<SignalIdx>& signals() const {
        return m_signals;
    }

    const std::vector<StatementIdx>& statements() const {
        return m_statements;
    }

    void addSignal(SignalIdx signal) {
        m_signals.push_back(signal);
    }

private:
    std::vector<SignalIdx> m_signals;
    std::vector<StatementIdx> m_statements;
};

struct Module : NodeBase {
public:
    Module(std::string_view name, Loc loc) : NodeBase(name, loc) {}

    void addSignal(SignalIdx signal) {
        m_signals.push_back(signal);
    }

    void addProcess(ProcessIdx proc) {
        m_processes.push_back(proc);
    }

    const std::vector<SignalIdx>& signals() const {
        return m_signals;
    }

    const std::vector<ProcessIdx>& processes() const {
        return m_processes;
    }

private:
    std::vector<SignalIdx> m_signals;
    std::vector<ProcessIdx> m_processes;
};

struct Ast : GenericAst<Signal, Process, Module, Expr, Type> {
    using GenericAst::GenericAst;

public:
    Ast() : m_top_module() {}

    const Module& getTopModule() const {
        return getNode(m_top_module);
    }

    void setTopModule(ModuleIdx top_module) {
        m_top_module = top_module;
    }

private:
    ModuleIdx m_top_module;
};

} // namespace cir
