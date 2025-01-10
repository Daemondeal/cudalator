#pragma once

#include "CIR_Base.h"
#include "CIR_GenericAst.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace cir {

struct Ast;

struct Scope;
struct Signal;
struct Process;
struct Module;
struct Type;
struct Statement;
struct Expr;
struct Constant;

using ScopeIdx = NodeIndex<Scope>;
using SignalIdx = NodeIndex<Signal>;
using ProcessIdx = NodeIndex<Process>;
using ModuleIdx = NodeIndex<Module>;
using TypeIdx = NodeIndex<Type>;
using StatementIdx = NodeIndex<Statement>;
using ExprIdx = NodeIndex<Expr>;
using ConstantIdx = NodeIndex<Constant>;

struct Ast : GenericAst<Signal, Process, Module, Expr, Statement, Type,
                        Constant, Scope> {
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
    Invalid,

    Bit,
    Logic,
    Integer,
    Int,
};

struct Type {
public:
    Type(TypeKind kind) : m_kind(kind) {}

    Type(TypeKind kind, TypeIdx subtype) : m_kind(kind), m_subtype(subtype) {}

    Type() = delete;
    Type(const Type& other) = delete;
    Type& operator=(const Type& other) = delete;

    Type(Type&& other)
        : m_kind(other.m_kind), m_ranges(std::move(other.m_ranges)),
          m_subtype(other.m_subtype) {}

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

    size_t size() const {
        if (m_kind == TypeKind::Invalid) {
            return 0;
        }

        if (m_kind == TypeKind::Int || m_kind == TypeKind::Integer) {
            return 32;
        }

        if (m_ranges.size() == 0) {
            return 1;
        }

        // CD_ASSERT_MSG(m_ranges.size() < 2, "more than 1 range is unimplemented");

        if (m_ranges.size() == 0) {
            auto range = firstRange();
            return range.right() - range.left() + 1;
        }

        return 0;
    }

private:
    TypeKind m_kind;
    std::vector<Range> m_ranges;
    TypeIdx m_subtype;
};

enum class SignalLifetime {
    Static,
    Automatic,
    Net,
};

struct Signal : NodeBase {
public:
    explicit Signal(std::string_view name, Loc loc, std::string_view full_name,
                    TypeIdx type, SignalLifetime lifetime)
        : NodeBase(name, loc), m_type(type), m_full_name(full_name),
          m_lifetime(lifetime) {}

    Signal() = delete;
    Signal(const Signal& other) = delete;
    Signal& operator=(Signal& other) = delete;

    Signal(Signal&& other) noexcept
        : NodeBase(std::move(other)), m_type(other.m_type),
          m_full_name(other.m_full_name), m_lifetime(other.m_lifetime) {}

    std::string_view fullName() const {
        return m_full_name;
    }
    TypeIdx type() const {
        return m_type;
    }

    SignalLifetime lifetime() const {
        return m_lifetime;
    }

private:
    TypeIdx m_type;

    std::string_view m_full_name;
    SignalLifetime m_lifetime;
};

struct Constant : NodeBase {
public:
    Constant(std::string_view name, Loc loc, uint32_t size)
        : NodeBase(name, loc), m_size(size) {}
    Constant(std::string_view name, Loc loc, uint32_t size, uint64_t value)
        : NodeBase(name, loc), m_size(size), m_value({value}) {}

    Constant() = delete;
    Constant(Constant& other) = delete;
    Constant& operator=(Constant& other) = delete;

    Constant(Constant&& other)
        : NodeBase(std::move(other)), m_size(other.m_size),
          m_value(std::move(other.m_value)) {}

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
    Invalid,

    Constant,

    // Unary:
    // (exprs[0]: operand)
    UnaryMinus,
    UnaryPlus,
    Not,
    BinaryNegation,
    ReductionAnd,
    ReductionNand,
    ReductionOr,
    ReductionNor,
    ReductionXor,
    ReductionXnor,
    Posedge,
    Negedge,

    // Binary:
    // (exprs[0]: lhs, exprs[1]: rhs)
    Subtraction,
    Division,
    Modulo,
    Equality,
    NotEquality,
    GreaterThan,
    GreaterThanEq,
    LessThan,
    LessThanEq,
    LeftShift,
    RightShift,
    Addition,
    Multiplication,
    LogicalAnd,
    LogicalOr,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseXnor,

    // exprs[0..len-1]: sub-exprs to concatenate
    Concatenation,

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

    Expr(std::string_view name, Loc loc, ExprKind kind, ExprIdx operand)
        : NodeBase(name, loc), m_kind(kind), m_exprs({operand}) {}

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

    Expr() = delete;
    Expr(const Expr& other) = delete;
    Expr& operator=(const Expr& other) = delete;

    Expr(Expr&& other)
        : NodeBase(std::move(other)), m_kind(other.m_kind),
          m_exprs(std::move(other.m_exprs)), m_constant(other.m_constant),
          m_signal(other.m_signal) {}

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

    // clang-format off
    bool isUnary() const {
        return (m_kind == ExprKind::UnaryMinus) ||
               (m_kind == ExprKind::UnaryPlus) ||
               (m_kind == ExprKind::Not) ||
               (m_kind == ExprKind::BinaryNegation) ||
               (m_kind == ExprKind::ReductionAnd) ||
               (m_kind == ExprKind::ReductionNand) ||
               (m_kind == ExprKind::ReductionOr) ||
               (m_kind == ExprKind::ReductionNor) ||
               (m_kind == ExprKind::ReductionXor) ||
               (m_kind == ExprKind::ReductionXnor) ||
               (m_kind == ExprKind::Posedge) ||
               (m_kind == ExprKind::Negedge);
    }

    bool isBinary() const {
        return (m_kind == ExprKind::Subtraction) ||
               (m_kind == ExprKind::Division) ||
               (m_kind == ExprKind::Modulo) ||
               (m_kind == ExprKind::Equality) ||
               (m_kind == ExprKind::NotEquality) ||
               (m_kind == ExprKind::GreaterThan) ||
               (m_kind == ExprKind::GreaterThanEq) ||
               (m_kind == ExprKind::LessThan) ||
               (m_kind == ExprKind::LessThanEq) ||
               (m_kind == ExprKind::LeftShift) ||
               (m_kind == ExprKind::RightShift) ||
               (m_kind == ExprKind::Addition) ||
               (m_kind == ExprKind::Multiplication) ||
               (m_kind == ExprKind::LogicalAnd) ||
               (m_kind == ExprKind::LogicalOr) ||
               (m_kind == ExprKind::BitwiseAnd) ||
               (m_kind == ExprKind::BitwiseOr) ||
               (m_kind == ExprKind::BitwiseXor) ||
               (m_kind == ExprKind::BitwiseXnor);
    }
    // clang-format on

private:
    ExprKind m_kind;

    std::vector<ExprIdx> m_exprs;

    ConstantIdx m_constant;

    SignalIdx m_signal;
};

enum class StatementKind {
    Invalid,

    // lhs: target, rhs: value
    NonBlockingAssignment,

    // lhs: target, rhs: value
    Assignment,

    // statements[0..len-1]: child statements in order, scope: variables scope
    Block,

    // lhs: condition, statements[0]: body
    If,

    // lhs: condition, statements[0]: body, statements[1]: else
    IfElse,

    // lhs: condition, statements[0]: body
    While,
    DoWhile,

    // lhs: condition, statements[0]: body
    Repeat,

    // lhs: condition, statements[0]: init, statements[1]: increment,
    // statements[2]: body
    For,

    // TODO: START
    Case,
    Foreach,
    // TODO: END

    // statements[0]: body
    Forever,

    // lhs: expr
    Return,

    // Nothing needed
    Break,
    Continue,
    Null,

};

struct Statement : NodeBase {
public:
    Statement(std::string_view name, Loc loc, StatementKind kind)
        : NodeBase(name, loc), m_kind(kind) {}

    Statement() = delete;
    Statement(const Statement& other) = delete;
    Statement& operator=(const Statement& other) = delete;

    Statement(Statement&& other)
        : NodeBase(std::move(other)), m_kind(other.m_kind), m_lhs(other.m_lhs),
          m_rhs(other.m_rhs), m_scope(other.m_scope),
          m_statements(std::move(other.m_statements)) {}

    void setScope(ScopeIdx scope) {
        m_scope = scope;
    }

    void setLhs(ExprIdx lhs) {
        m_lhs = lhs;
    }

    void setRhs(ExprIdx rhs) {
        m_rhs = rhs;
    }

    StatementKind kind() const {
        return m_kind;
    }

    ExprIdx lhs() const {
        return m_lhs;
    }

    ExprIdx rhs() const {
        return m_rhs;
    }

    ScopeIdx scope() const {
        return m_scope;
    }

    StatementIdx body() const {
        return m_statements[0];
    }

    const std::vector<StatementIdx>& statements() const {
        return m_statements;
    }

    StatementIdx statement(size_t idx) const {
        return m_statements[idx];
    }

    void addStatement(StatementIdx statement) {
        m_statements.push_back(statement);
    }

private:
    StatementKind m_kind;

    ExprIdx m_lhs;
    ExprIdx m_rhs;

    ScopeIdx m_scope;

    std::vector<StatementIdx> m_statements;
};

enum class SensitivityKind : uint8_t { OnChange, Posedge, Negedge };

struct SensitivityListElement {
public:
    SensitivityListElement(SignalIdx signal, SensitivityKind kind)
        : kind(kind), signal(signal) {}

    SensitivityKind kind;
    SignalIdx signal;
};

struct Process : NodeBase {
public:
    Process(std::string_view name, Loc loc, StatementIdx statement)
        : NodeBase(name, loc), m_statement(statement) {}

    Process() = delete;
    Process(const Process& other) = delete;
    Process& operator=(const Process& other) = delete;

    Process(Process&& other)
        : NodeBase(std::move(other)),
          m_sensitivity_list(std::move(other.m_sensitivity_list)),
          m_statement(other.m_statement),
          m_should_populate_sensitivity_list(
              other.m_should_populate_sensitivity_list) {}

    const std::vector<SensitivityListElement>& sensitivityList() const {
        return m_sensitivity_list;
    }

    StatementIdx statement() const {
        return m_statement;
    }

    void addToSensitivityList(SignalIdx signal, SensitivityKind kind) {
        m_sensitivity_list.emplace_back(signal, kind);
    }

    void
    setSensitivityList(std::vector<SensitivityListElement>&& sensitivityList) {
        m_sensitivity_list = std::move(sensitivityList);
    }

    void setShouldPopulateSensitivityList(bool val) {
        m_should_populate_sensitivity_list = val;
    }

    bool shouldPopulateSensitivityList() const {
        return m_should_populate_sensitivity_list;
    }

private:
    std::vector<SensitivityListElement> m_sensitivity_list;
    StatementIdx m_statement;

    bool m_should_populate_sensitivity_list = false;
};

enum class SignalDirection {
    Input,
    Output,
    Inout,
    Invalid,
};

struct ModulePort {
public:
    ModulePort(SignalIdx signal, SignalDirection direction)
        : signal(signal), direction(direction) {}

    SignalIdx signal;
    SignalDirection direction;
};

struct Module : NodeBase {
public:
    Module(std::string_view name, Loc loc, ScopeIdx scope)
        : NodeBase(name, loc), m_scope(scope) {}

    Module() = delete;
    Module(const Module& other) = delete;
    Module& operator=(const Module& other) = delete;

    Module(Module&& other)
        : NodeBase(std::move(other)), m_scope(other.m_scope),
          m_ports(std::move(other.m_ports)),
          m_processes(std::move(other.m_processes)) {}

    void addPort(ModulePort port) {
        m_ports.push_back(port);
    }

    void addProcess(ProcessIdx proc) {
        m_processes.push_back(proc);
    }

    ScopeIdx scope() const {
        return m_scope;
    }

    const std::vector<ModulePort>& ports() const {
        return m_ports;
    }

    const std::vector<ProcessIdx>& processes() const {
        return m_processes;
    }

private:
    ScopeIdx m_scope;

    std::vector<ModulePort> m_ports;
    std::vector<ProcessIdx> m_processes;
};

struct Scope : NodeBase {
public:
    Scope(std::string_view name, Loc loc, ScopeIdx parent)
        : NodeBase(name, loc), m_parent(parent), m_signals({}) {}

    Scope() = delete;
    Scope(const Scope& other) = delete;
    Scope& operator=(const Scope& other) = delete;

    Scope(Scope&& other)
        : NodeBase(std::move(other)), m_parent(other.m_parent),
          m_signals(std::move(other.m_signals)) {}

    ScopeIdx parent() const {
        return m_parent;
    }

    const std::vector<SignalIdx>& signals() const {
        return m_signals;
    }

    void addSignal(SignalIdx signal) {
        m_signals.push_back(signal);
    }

    SignalIdx findSignalByName(Ast& ast, std::string_view full_name) const {
        for (auto idx : m_signals) {
            auto& signal = ast.getNode(idx);
            if (signal.fullName() == full_name) {
                return idx;
            }
        }

        if (m_parent.isValid()) {
            return ast.getNode(m_parent).findSignalByName(ast, full_name);
        }

        return SignalIdx::null();
    }

private:
    ScopeIdx m_parent;

    std::vector<SignalIdx> m_signals;
};

} // namespace cir
