use paste::paste;

#[derive(Copy, Clone)]
pub struct ScopeIdx(usize);

#[derive(Copy, Clone)]
pub struct SignalIdx(usize);

#[derive(Copy, Clone)]
pub struct ProcessIdx(usize);

#[derive(Copy, Clone)]
pub struct ModuleIdx(usize);

#[derive(Copy, Clone)]
pub struct TypeIdx(usize);

#[derive(Copy, Clone)]
pub struct StatementIdx(usize);

#[derive(Copy, Clone)]
pub struct ExprIdx(usize);

#[derive(Copy, Clone)]
pub struct ConstantIdx(usize);

// TODO: Maybe this needs another name
#[derive(Clone)]
pub struct Token {
    pub name: String,
    pub line: u32,
}

impl Token {
    pub fn dummy() -> Self {
        Self {
            name: "".to_owned(),
            line: 0,
        }
    }
}

pub enum PortDirection {
    Input,
    Output,
    Inout,
    Invalid,
}

pub struct ModulePort {
    pub signal: SignalIdx,
    pub direction: PortDirection,
}

pub struct Module {
    pub token: Token,

    pub scope: ScopeIdx,
    pub ports: Vec<ModulePort>,
    pub processes: Vec<ProcessIdx>,
}

pub struct Scope {
    pub token: Token,

    pub parent: Option<ScopeIdx>,
    pub signals: Vec<SignalIdx>,
}

impl Scope {
    pub fn find_signal_by_name(&self, ast: &Ast, full_name: &str) -> Option<SignalIdx> {
        for idx in &self.signals {
            if ast.get_signal(*idx).full_name == full_name {
                return Some(*idx);
            }
        }

        match self.parent {
            Some(parent) => ast.get_scope(parent).find_signal_by_name(ast, full_name),
            None => None,
        }
    }
}

pub enum SensitivtyKind {
    OnChange,
    Posedge,
    Negedge,
}

pub struct Process {
    pub token: Token,
    pub statement: StatementIdx,

    pub sensitivity_list: Vec<(SensitivtyKind, SignalIdx)>,
    pub should_populate_sensitivity_list: bool,
}

pub enum StatementKind {
    Invalid,

    Assignment {
        lhs: ExprIdx,
        rhs: ExprIdx,
        blocking: bool,
    },
    Block {
        statements: Vec<StatementIdx>,
    },
    ScopedBlock {
        statements: Vec<StatementIdx>,
        scope: ScopeIdx,
    },
    If {
        condition: ExprIdx,
        body: StatementIdx,
    },
    IfElse {
        condition: ExprIdx,
        body: StatementIdx,
        else_: StatementIdx,
    },
    While {
        condition: ExprIdx,
        body: StatementIdx,
    },
    DoWhile {
        condition: ExprIdx,
        body: StatementIdx,
    },

    Repeat {
        condition: ExprIdx,
        body: StatementIdx,
    },

    For {
        condition: ExprIdx,
        init: StatementIdx,
        increment: StatementIdx,
        body: StatementIdx,
        scope: ScopeIdx,
    },

    // TODO
    Case,
    Foreach,

    Forever {
        body: StatementIdx,
    },

    Return {
        expr: ExprIdx,
    },

    Break,
    Continue,
    Null,
}

pub struct Statement {
    pub token: Token,
    pub kind: StatementKind,
}

pub enum UnaryOperator {
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
}

impl UnaryOperator {
    pub fn to_str(&self) -> &'static str {
        match self {
            UnaryOperator::UnaryMinus => "-",
            UnaryOperator::UnaryPlus => "+",
            UnaryOperator::Not => "!",
            UnaryOperator::BinaryNegation => "~",
            UnaryOperator::ReductionAnd => "red_and",
            UnaryOperator::ReductionNand => "red_nand",
            UnaryOperator::ReductionOr => "red_or",
            UnaryOperator::ReductionNor => "red_nor",
            UnaryOperator::ReductionXor => "red_xor",
            UnaryOperator::ReductionXnor => "red_xnor",
            UnaryOperator::Posedge => "posedge",
            UnaryOperator::Negedge => "negedge",
        }
    }
}

pub enum BinaryOperator {
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
}

impl BinaryOperator {
    pub fn to_str(&self) -> &'static str {
        match self {
            BinaryOperator::Subtraction => "-",
            BinaryOperator::Division => "/",
            BinaryOperator::Modulo => "%",
            BinaryOperator::Equality => "==",
            BinaryOperator::NotEquality => "!=",
            BinaryOperator::GreaterThan => ">",
            BinaryOperator::GreaterThanEq => ">=",
            BinaryOperator::LessThan => "<",
            BinaryOperator::LessThanEq => "<=",
            BinaryOperator::LeftShift => "<<",
            BinaryOperator::RightShift => ">>",
            BinaryOperator::Addition => "+",
            BinaryOperator::Multiplication => "*",
            BinaryOperator::LogicalAnd => "and",
            BinaryOperator::LogicalOr => "or",
            BinaryOperator::BitwiseAnd => "bw_and",
            BinaryOperator::BitwiseOr => "bw_or",
            BinaryOperator::BitwiseXor => "bw_xor",
            BinaryOperator::BitwiseXnor => "bw_xnor",
        }
    }
}

pub enum ExprKind {
    Invalid,
    Constant {
        constant: ConstantIdx
    },
    Unary {
        op: UnaryOperator,
        expr: ExprIdx,
    },
    Binary {
        op: BinaryOperator,
        lhs: ExprIdx,
        rhs: ExprIdx,
    },

    Concatenation {
        exprs: Vec<ExprIdx>,
    },

    PartSelect {
        lhs: ExprIdx,
        rhs: ExprIdx,
        target: SignalIdx,
    },
    BitSelect {
        expr: ExprIdx,
        target: SignalIdx,
    },
    SignalRef {
        signal: SignalIdx,
    },
}

pub enum SignalLifetime {
    Static,
    Automatic,
    Net,
}

pub struct Signal {
    pub token: Token,

    pub typ: TypeIdx,
    pub lifetime: SignalLifetime,
    pub full_name: String,
}

pub struct Expr {
    pub token: Token,
    pub kind: ExprKind,
}

pub enum TypeKind {
    Invalid,

    Integer,
    Int,
    Bit(Option<Range>),
    Logic(Option<Range>),
}

pub struct Type {
    pub token: Token,

    pub kind: TypeKind,
    pub is_signed: bool,
}

#[derive(Clone)]
pub struct Range {
    pub left: u32,
    pub right: u32,
}

pub enum ConstantKind {
    Integer(i64),
    UnsignedInteger(u64),
    Value { vals: Vec<u32> },
    Invalid,
}

// TODO: This needs to be done properly
pub struct Constant {
    pub token: Token,

    pub size: u32,
    pub kind: ConstantKind,
}

pub struct Ast {
    pub modules: Vec<Module>,
    pub processes: Vec<Process>,
    pub statements: Vec<Statement>,
    pub exprs: Vec<Expr>,
    pub signals: Vec<Signal>,
    pub types: Vec<Type>,
    pub constants: Vec<Constant>,
    pub scopes: Vec<Scope>,

    pub top_module: Option<ModuleIdx>,
}

// TODO: Add documentation for how this works
macro_rules! add_array {
    ($array:ident, $name:ident, $type:ident, $idx:ident) => {
        paste! {
            pub fn [<get_ $name>](&self, idx: $idx) -> &$type {
                &self.$array[idx.0]
            }

            pub fn [<get_ $name _mut>](&mut self, idx: $idx) -> &mut $type {
                &mut self.$array[idx.0]
            }

            pub fn [<add_ $name>](&mut self, $name: $type) -> $idx {
                self.$array.push($name);

                $idx(self.$array.len() - 1)
            }
        }
    };
}

// TODO: Add documentation for how this works
impl Ast {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    add_array!(modules, module, Module, ModuleIdx);
    add_array!(scopes, scope, Scope, ScopeIdx);
    add_array!(signals, signal, Signal, SignalIdx);
    add_array!(processes, process, Process, ProcessIdx);
    add_array!(types, typ, Type, TypeIdx);
    add_array!(statements, statement, Statement, StatementIdx);
    add_array!(exprs, expr, Expr, ExprIdx);
    add_array!(constants, constant, Constant, ConstantIdx);
}

impl Default for Ast {
    fn default() -> Self {
        Self {
            modules: vec![],
            processes: vec![],
            statements: vec![],
            exprs: vec![],
            signals: vec![],
            types: vec![],
            constants: vec![],
            scopes: vec![],
            top_module: None,
        }
    }
}
