use paste::paste;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ScopeIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct SignalIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ProcessIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ModuleIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct StatementIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExprIdx(usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConstantIdx(usize);

// FIXME: We should probably have a better way of having unique ids for processes
impl ProcessIdx {
    pub fn get_idx(&self) -> u32 {
        self.0 as u32
    }
}

impl SignalIdx {
    pub fn get_idx(&self) -> u32 {
        self.0 as u32
    }

    // NOTE: This is somewhat dangerous, only use this if you are iterating over the signal array.
    // TODO: Make an iterator for the signal array that generates the idx directly so we don't have
    //       to expose this.
    pub fn from_idx(x: u32) -> Self {
        Self(x as usize)
    }
}

// TODO: Maybe this needs another name
#[derive(Clone, Debug)]
pub struct Token {
    pub name: String,
    pub file: String,
    pub line: u32,
}

impl Token {
    pub fn dummy() -> Self {
        Self {
            line: 0,
            file: "".to_string(),
            name: "".to_string(),
        }
    }

    pub fn named_dummy(name: impl Into<String>) -> Self {
        Self {
            line: 0,
            file: "".to_string(),
            name: name.into(),
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
    pub is_top: bool,
}

impl Scope {
    pub fn find_signal(&self, ast: &Ast, full_name: &str) -> Option<SignalIdx> {
        for idx in &self.signals {
            if ast.get_signal(*idx).full_name == full_name {
                return Some(*idx);
            }
        }

        None
    }

    pub fn find_signal_recursively(&self, ast: &Ast, full_name: &str) -> Option<SignalIdx> {
        match self.find_signal(ast, full_name) {
            Some(idx) => Some(idx),
            None => match self.parent {
                Some(parent) => ast
                    .get_scope(parent)
                    .find_signal_recursively(ast, full_name),
                None => None,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SensitivtyKind {
    OnChange,
    Posedge,
    Negedge,
}

pub struct Block {
    pub scope: ScopeIdx,
    pub statements: Vec<StatementIdx>,
}

pub struct Process {
    pub token: Token,
    // pub statement: StatementIdx,
    pub scope: ScopeIdx,
    pub statements: Vec<StatementIdx>,

    pub sensitivity_list: Vec<(SensitivtyKind, SignalIdx)>,
    pub should_populate_sensitivity_list: bool,
}

pub enum SelectKind {
    None,
    Bit(ExprIdx),
    Parts { lhs: ExprIdx, rhs: ExprIdx },
}

pub enum StatementKind {
    Invalid,

    Assignment {
        lhs: SignalIdx,
        rhs: ExprIdx,
        select: SelectKind,
    },
    Block {
        statements: Vec<StatementIdx>,
        scope: ScopeIdx,
    },
    If {
        condition: ExprIdx,
        body: StatementIdx,
        else_: Option<StatementIdx>,
    },
    While {
        condition: ExprIdx,
        body: StatementIdx,
    },
    DoWhile {
        condition: ExprIdx,
        body: StatementIdx,
    },

    // TODO
    Case,
    Foreach,

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

#[derive(Clone, Copy, Debug)]
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

    // LRM 11.6.1 Table 11-21
    pub fn get_size(&self, length_i: usize) -> usize {
        match self {
            UnaryOperator::UnaryMinus
            | UnaryOperator::UnaryPlus
            | UnaryOperator::BinaryNegation => length_i,

            UnaryOperator::Not
            | UnaryOperator::ReductionAnd
            | UnaryOperator::ReductionNand
            | UnaryOperator::ReductionOr
            | UnaryOperator::ReductionNor
            | UnaryOperator::ReductionXor
            | UnaryOperator::ReductionXnor => 1,

            UnaryOperator::Posedge | UnaryOperator::Negedge => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
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
    Power,
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
            BinaryOperator::Power => "**",
            BinaryOperator::LogicalAnd => "and",
            BinaryOperator::LogicalOr => "or",
            BinaryOperator::BitwiseAnd => "bw_and",
            BinaryOperator::BitwiseOr => "bw_or",
            BinaryOperator::BitwiseXor => "bw_xor",
            BinaryOperator::BitwiseXnor => "bw_xnor",
        }
    }

    // LRM 11.6.1 Table 11-21
    pub fn get_size(&self, length_i: usize, length_j: usize) -> usize {
        match self {
            BinaryOperator::Addition
            | BinaryOperator::Subtraction
            | BinaryOperator::Multiplication
            | BinaryOperator::Division
            | BinaryOperator::Modulo
            | BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseOr
            | BinaryOperator::BitwiseXor
            | BinaryOperator::BitwiseXnor => std::cmp::max(length_i, length_j),

            BinaryOperator::Equality
            | BinaryOperator::NotEquality
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanEq
            | BinaryOperator::LessThan
            | BinaryOperator::LessThanEq => 1,

            // TODO: Operands are sized to max(L(i),L(j))
            BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => 1,

            BinaryOperator::LeftShift | BinaryOperator::RightShift | BinaryOperator::Power => {
                length_i
            }
        }
    }
}

#[derive(Debug)]
pub enum ExprKind {
    Invalid,
    Constant {
        constant: ConstantIdx,
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

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
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
    pub scope: ScopeIdx,

    pub is_in_top_interface: bool,
}

impl Signal {
    pub fn size(&self, ast: &Ast) -> usize {
        let typ = ast.get_typ(self.typ);
        return typ.size;
    }
}

pub struct Expr {
    pub token: Token,
    pub kind: ExprKind,
    pub size: usize,
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
    pub size: usize,
}

#[derive(Clone)]
pub struct Range {
    pub left: u32,
    pub right: u32,
}

impl Range {
    pub fn size(&self) -> u32 {
        if self.left > self.right {
            1 + self.left - self.right
        } else {
            1 + self.right - self.left
        }
    }
}

#[derive(Debug)]
pub enum ConstantKind {
    Integer(i64),
    UnsignedInteger(u64),
    Value { vals: Vec<u32> },
    AllZero,
    AllOnes,
    Invalid,
}

// TODO: This needs to be done properly
#[derive(Debug)]
pub struct Constant {
    pub token: Token,

    pub size: usize,
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
