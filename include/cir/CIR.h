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
struct Constant;

using SignalIdx = NodeIndex<Signal>;
using ProcessIdx = NodeIndex<Process>;
using ModuleIdx = NodeIndex<Module>;
using TypeIdx = NodeIndex<Type>;
using StatementIdx = NodeIndex<Statement>;
using ExprIdx = NodeIndex<Expr>;
using ConstantIdx = NodeIndex<Constant>;

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
    explicit Signal(std::string_view name, Loc loc, std::string_view full_name,
                    TypeIdx type, SignalDirection kind)
        : NodeBase(name, loc), m_full_name(full_name), m_type(type),
          m_kind(kind) {}

    std::string_view fullName() {
        return m_full_name;
    }
    TypeIdx type() const {
        return m_type;
    }

    SignalDirection kind() const {
        return m_kind;
    }

private:
    TypeIdx m_type;

    std::string_view m_full_name;
    SignalDirection m_kind;
};

struct Constant : NodeBase {
public:
    Constant(std::string_view name, Loc loc, uint32_t size)
        : NodeBase(name, loc), m_size(size) {}
    Constant(std::string_view name, Loc loc, uint32_t size, uint64_t value)
        : NodeBase(name, loc), m_size(size), m_value({value}) {}

    // TODO: There has to be a better way to do this
    void addValue(uint32_t val) {
        m_value.push_back(val);
    }

    // TODO: This is bad
    uint64_t value() {
        return m_value[0];
    }

private:
    uint32_t m_size;
    std::vector<uint64_t> m_value;
};

enum class ExprKind {
    Constant,

    // exprs[0]: lhs, exprs[1]: rhs
    Addition,

    // exprs[0]: lhs, exprs[1]: rhs, signal: target
    PartSelect,
    // exprs[0]: bit number, signal: target
    BitSelect,
    // signal: target
    SignalRef,

};

struct Expr : NodeBase {
public:
    Expr(std::string_view name, Loc loc, ExprKind kind, ExprIdx lhs,
         ExprIdx rhs)
        : NodeBase(name, loc), m_kind(kind), m_exprs({lhs, rhs}) {}

    Expr(std::string_view name, Loc loc, ExprKind kind, ExprIdx lhs,
         ExprIdx rhs, SignalIdx signal)
        : NodeBase(name, loc), m_kind(kind), m_exprs({lhs, rhs}),
          m_signal(signal) {}

    Expr(std::string_view name, Loc loc, ExprKind kind, SignalIdx signal)
        : NodeBase(name, loc), m_kind(kind), m_signal(signal) {}

    Expr(std::string_view name, Loc loc, ExprKind kind, ConstantIdx constant)
        : NodeBase(name, loc), m_kind(kind), m_constant(constant) {}

    Expr(std::string_view name, Loc loc, ExprKind kind)
        : NodeBase(name, loc), m_kind(kind) {}

    ExprIdx lhs() const {
        return m_exprs[0];
    }

    ExprIdx rhs() const {
        return m_exprs[1];
    }

    ExprKind kind() const {
        return m_kind;
    }

    SignalIdx signal() const {
        return m_signal;
    }

    ConstantIdx constant() const {
        return m_constant;
    }

    ExprIdx expr(int idx) const {
        return m_exprs[idx];
    }

    const std::vector<ExprIdx>& exprs() const {
        return m_exprs;
    }

    void addExpr(ExprIdx expr) {
        m_exprs.push_back(expr);
    }

private:
    ExprKind m_kind;

    std::vector<ExprIdx> m_exprs;

    ConstantIdx m_constant;

    SignalIdx m_signal;
};

enum class StatementKind {
    Assignment,
};

struct Statement : NodeBase {
public:
    Statement(std::string_view name, Loc loc, StatementKind kind)
        : NodeBase(name, loc), m_kind(kind) {}

    Statement(std::string_view name, Loc loc, StatementKind kind, ExprIdx lhs,
              ExprIdx rhs)
        : NodeBase(name, loc), m_kind(kind), m_lhs(lhs), m_rhs(rhs) {}

    StatementKind kind() const {
        return m_kind;
    }

    ExprIdx lhs() const {
        return m_lhs;
    }

    ExprIdx rhs() const {
        return m_rhs;
    }

    const std::vector<StatementIdx>& statements() const {
        return m_statements;
    }

    void addStatement(StatementIdx statement) {
        m_statements.push_back(statement);
    }

private:
    StatementKind m_kind;

    ExprIdx m_lhs;
    ExprIdx m_rhs;

    std::vector<StatementIdx> m_statements;
};

struct Process : NodeBase {
public:
    Process(std::string_view name, Loc loc, StatementIdx statement)
        : NodeBase(name, loc), m_statement(statement) {}

    const std::vector<SignalIdx>& sensitivityList() const {
        return m_sensitivity_list;
    }

    StatementIdx statement() const {
        return m_statement;
    }

    void addToSensitivityList(SignalIdx signal) {
        m_sensitivity_list.push_back(signal);
    }

    void setShouldPopulateSensitivityList(bool val) {
        m_should_populate_sensitivity_list = val;
    }

    bool shouldPopulateSensitivityList() const {
        return m_should_populate_sensitivity_list;
    }

private:
    std::vector<SignalIdx> m_sensitivity_list;
    StatementIdx m_statement;

    bool m_should_populate_sensitivity_list = false;
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

struct Ast
    : GenericAst<Signal, Process, Module, Expr, Statement, Type, Constant> {
    using GenericAst::GenericAst;

public:
    Ast() : m_top_module() {}

    const Module& getTopModule() const {
        return getNode(m_top_module);
    }

    void setTopModule(ModuleIdx top_module) {
        m_top_module = top_module;
    }

    SignalIdx findSignal(std::string_view full_name) {
        auto signal_vec = getNodeVector<Signal>();
        return signal_vec.findByFullName(full_name);
    }

private:
    ModuleIdx m_top_module;
};

} // namespace cir
