use core::fmt;
use std::io::Write;

use color_eyre::Result;

use crate::cir::{
    Ast, BinaryOperator, ConstantIdx, ConstantKind, ExprIdx, ExprKind, ModuleIdx, ProcessIdx,
    ScopeIdx, SignalIdx, StatementIdx, StatementKind, TypeIdx, TypeKind, UnaryOperator,
};

pub enum CodegenTarget {
    CPU,
    CUDA,
}

enum OperatorType {
    Infix(&'static str),
    Function(&'static str),
}

struct CppEmitter<'a, W: Write> {
    writer: &'a mut W,
    indent_level: i32,
}

impl<'a, W: Write> CppEmitter<'a, W> {
    pub fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            indent_level: 0,
        }
    }

    // TODO: This is probably not the right way to do this
    pub fn emit(&mut self, fmt: fmt::Arguments<'_>) -> Result<()> {
        write!(self.writer, "{fmt}")?;
        Ok(())
    }

    pub fn line_start(&mut self) -> Result<()> {
        for _ in 0..self.indent_level {
            write!(self.writer, "  ")?;
        }
        Ok(())
    }

    pub fn line_end(&mut self) -> Result<()> {
        writeln!(self.writer)?;
        Ok(())
    }

    pub fn line_end_semicolon(&mut self) -> Result<()> {
        writeln!(self.writer, ";")?;
        Ok(())
    }

    pub fn block_start(&mut self) -> Result<()> {
        self.line_start()?;
        write!(self.writer, "{{")?;
        self.line_end()?;
        self.indent_level += 1;
        Ok(())
    }

    pub fn block_end(&mut self) -> Result<()> {
        self.indent_level -= 1;
        self.line_start()?;
        write!(self.writer, "}}")?;
        self.line_end()?;
        Ok(())
    }

    pub fn block_end_semicolon(&mut self) -> Result<()> {
        self.indent_level -= 1;
        self.line_start()?;
        write!(self.writer, "}};")?;
        self.line_end()?;
        Ok(())
    }

    pub fn emit_empty_line(&mut self) -> Result<()> {
        write!(self.writer, "\n")?;
        Ok(())
    }
}

macro_rules! emit {
    ($emitter:expr, $($args:tt)*) => {
        $emitter.emit(format_args!($($args)*))
    };
}

// TODO: Maybe we should think more about this
fn clean_ident(ident: &str) -> String {
    ident.replace("@", "__").replace(".", "__")
}

struct Codegen<'a> {
    ast: &'a Ast,
    top_module_idx: ModuleIdx,
    state_struct_name: String,
    target: CodegenTarget,
}

impl<'a> Codegen<'a> {
    pub fn new(ast: &'a Ast, target: CodegenTarget) -> Self {
        let top_module_idx = ast.top_module.expect("No top module found");
        let state_struct_name = clean_ident(&ast.get_module(top_module_idx).token.name);

        Self {
            ast,
            top_module_idx,
            state_struct_name,
            target,
        }
    }

    pub fn codegen<W: Write>(
        &self,
        source: &mut CppEmitter<'a, W>,
        header: &mut CppEmitter<'a, W>,
    ) -> Result<()> {
        self.codegen_struct(header)?;

        let top_module = self.ast.get_module(self.top_module_idx);
        for process in &top_module.processes {
            self.codegen_process_body(source, *process)?;
            source.emit_empty_line()?;
        }

        Ok(())
    }

    fn codegen_struct<W: Write>(&self, header: &mut CppEmitter<'a, W>) -> Result<()> {
        let top_module = self.ast.get_module(self.top_module_idx);
        let top_scope = self.ast.get_scope(top_module.scope);

        header.line_start()?;
        emit!(header, "struct {}", self.state_struct_name)?;
        header.line_end()?;

        header.block_start()?;
        for signal in &top_scope.signals {
            self.codegen_signal_declaration(header, *signal)?;
        }
        header.block_end_semicolon()?;

        Ok(())
    }

    fn codegen_process_body<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        process: ProcessIdx,
    ) -> Result<()> {
        let ast_process = self.ast.get_process(process);

        file.line_start()?;

        match self.target {
            CodegenTarget::CPU => {
                emit!(file, "void ")?;
            }
            CodegenTarget::CUDA => {
                emit!(file, "__global__ void ")?;
            }
        }
        emit!(
            file,
            "process__{} ({} *prev, {} *next, size_t len)",
            process.get_idx(),
            self.state_struct_name,
            self.state_struct_name
        )?;

        file.line_end()?;

        file.block_start()?;

        file.line_start()?;
        match self.target {
            CodegenTarget::CPU => {
                emit!(file, "int tid = 0")?;
            }
            CodegenTarget::CUDA => {
                emit!(file, "int tid = blockIdx.x * blockSize.x + threadIdx.x")?;
            }
        }
        file.line_end_semicolon()?;

        let stmt = self.ast.get_statement(ast_process.statement);

        match &stmt.kind {
            StatementKind::Block { statements } => {
                for statement in statements {
                    self.codegen_statement(file, *statement)?;
                }
            }

            _ => {
                self.codegen_statement(file, ast_process.statement)?;
            }
        }

        file.block_end()?;

        Ok(())
    }

    fn codegen_scope<W: Write>(&self, file: &mut CppEmitter<'a, W>, scope: ScopeIdx) -> Result<()> {
        Ok(())
    }

    fn codegen_statement<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        statement: StatementIdx,
    ) -> Result<()> {
        let statement = self.ast.get_statement(statement);

        match &statement.kind {
            StatementKind::Assignment { lhs, rhs, blocking } => {
                file.line_start()?;
                self.codegen_expr(file, *lhs, false)?;
                emit!(file, " = ")?;
                self.codegen_expr(file, *rhs, true)?;
                file.line_end_semicolon()?;

                if !blocking {
                    file.line_start()?;
                    self.codegen_expr(file, *lhs, true)?;
                    emit!(file, " = ")?;
                    self.codegen_expr(file, *lhs, false)?;
                    file.line_end_semicolon()?;
                }
            }
            StatementKind::Block { statements } => {
                file.block_start()?;

                for statement in statements {
                    self.codegen_statement(file, *statement)?;
                }

                file.block_end()?;
            }
            StatementKind::ScopedBlock { statements, scope } => {
                file.block_start()?;

                self.codegen_scope(file, *scope)?;

                for statement in statements {
                    self.codegen_statement(file, *statement)?;
                }

                file.block_end()?;
            }
            StatementKind::If { condition, body } => {
                file.line_start()?;
                emit!(file, "if (")?;
                self.codegen_expr(file, *condition, true)?;
                emit!(file, ")")?;
                file.line_end()?;

                self.codegen_statement(file, *body)?;
            }
            StatementKind::IfElse {
                condition,
                body,
                else_,
            } => {
                file.line_start()?;
                emit!(file, "if (")?;
                self.codegen_expr(file, *condition, true)?;
                emit!(file, ")")?;
                file.line_end()?;

                self.codegen_statement(file, *body)?;

                file.line_start()?;
                emit!(file, "else")?;
                file.line_end()?;

                self.codegen_statement(file, *else_)?;
            }
            StatementKind::While { condition, body } => {
                file.line_start()?;
                emit!(file, "while (")?;
                self.codegen_expr(file, *condition, true)?;
                emit!(file, ")")?;
                file.line_end()?;

                self.codegen_statement(file, *body)?;
            }
            StatementKind::DoWhile { condition, body } => {
                file.line_start()?;
                emit!(file, "do")?;
                file.line_end()?;

                self.codegen_statement(file, *body)?;

                file.line_start()?;
                emit!(file, "while (")?;
                self.codegen_expr(file, *condition, true)?;
                emit!(file, ")")?;
                file.line_end_semicolon()?;
            }
            StatementKind::Repeat { condition: _, body: _ } => todo!("codegen Repeat"),
            StatementKind::For {
                condition: _,
                init: _,
                increment: _,
                body: _,
                scope: _,
            } => todo!("codegen For"),
            StatementKind::Case => todo!("codegen Case"),
            StatementKind::Foreach => todo!("codegen Foreach"),
            StatementKind::Forever { body } => {
                file.line_start()?;
                emit!(file, "while (true)")?;
                file.line_end()?;

                self.codegen_statement(file, *body)?;
            }
            StatementKind::Return { expr } => {
                file.line_start()?;
                emit!(file, "return ")?;
                self.codegen_expr(file, *expr, true)?;
                file.line_end_semicolon()?;
            }
            StatementKind::Break => {
                file.line_start()?;
                emit!(file, "break")?;
                file.line_end_semicolon()?;
            }
            StatementKind::Continue => {
                file.line_start()?;
                emit!(file, "continue")?;
                file.line_end_semicolon()?;
            }
            StatementKind::Invalid => unreachable!(),
            StatementKind::Null => {}
        };

        Ok(())
    }

    // FIXME: This will not work for larger constant, make this better.
    fn codegen_constant<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        constant: ConstantIdx,
    ) -> Result<()> {
        let constant = self.ast.get_constant(constant);

        match &constant.kind {
            ConstantKind::Integer(int) => emit!(file, "{}", int)?,
            ConstantKind::UnsignedInteger(int) => emit!(file, "{}u", int)?,
            ConstantKind::Value { vals } => {
                assert!(!vals.is_empty());
                if vals.len() > 1 {
                    todo!("codegen constant bigger than 32 bits");
                }

                emit!(file, "make_bit<{}>({})", constant.size, vals[0])?;
            }
            ConstantKind::Invalid => unreachable!(),
        };

        Ok(())
    }

    fn codegen_expr<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        expr: ExprIdx,
        is_prev: bool,
    ) -> Result<()> {
        let expr = self.ast.get_expr(expr);

        match &expr.kind {
            ExprKind::Constant { constant } => self.codegen_constant(file, *constant)?,
            ExprKind::Unary { op, expr } => {
                let op_equivalent = match op {
                    UnaryOperator::UnaryMinus => OperatorType::Infix("-"),
                    UnaryOperator::UnaryPlus => OperatorType::Infix("+"),
                    UnaryOperator::Not => todo!("codegen UnaryNot"),
                    UnaryOperator::BinaryNegation => todo!("codegen BinaryNegation"),
                    UnaryOperator::ReductionAnd => todo!("codegen ReductionAnd"),
                    UnaryOperator::ReductionNand => todo!("codegen ReductionNand"),
                    UnaryOperator::ReductionOr => todo!("codegen ReductionOr"),
                    UnaryOperator::ReductionNor => todo!("codegen ReductionNor"),
                    UnaryOperator::ReductionXor => todo!("codegen ReductionXor"),
                    UnaryOperator::ReductionXnor => todo!("codegen ReductionXnor"),
                    UnaryOperator::Posedge => unreachable!("should not codegen posedge"),
                    UnaryOperator::Negedge => unreachable!("should not codegen negedge"),
                };

                match op_equivalent {
                    OperatorType::Infix(name) => {
                        emit!(file, "{}", name)?;
                        self.codegen_expr(file, *expr, is_prev)?;
                    }

                    OperatorType::Function(name) => {
                        self.codegen_expr(file, *expr, is_prev)?;
                        emit!(file, ".{}()", name)?;
                    }
                };
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let op_equivalent = match op {
                    BinaryOperator::Subtraction => OperatorType::Infix("-"),
                    BinaryOperator::Division => OperatorType::Infix("/"),
                    BinaryOperator::Modulo => OperatorType::Infix("%"),
                    BinaryOperator::Equality => OperatorType::Infix("=="),
                    BinaryOperator::NotEquality => OperatorType::Infix("!="),
                    BinaryOperator::GreaterThan => OperatorType::Infix(">"),
                    BinaryOperator::GreaterThanEq => OperatorType::Infix(">="),
                    BinaryOperator::LessThan => OperatorType::Infix("<"),
                    BinaryOperator::LessThanEq => OperatorType::Infix("<="),
                    BinaryOperator::LeftShift => OperatorType::Infix("<<"),
                    BinaryOperator::RightShift => OperatorType::Infix(">>"),
                    BinaryOperator::Addition => OperatorType::Infix("+"),
                    BinaryOperator::Multiplication => OperatorType::Infix("*"),
                    BinaryOperator::Power => OperatorType::Function("pow"),
                    BinaryOperator::LogicalAnd => todo!("codegen LogicalAnd"),
                    BinaryOperator::LogicalOr => todo!("codegen LogicalOr"),
                    BinaryOperator::BitwiseAnd => OperatorType::Infix("&"),
                    BinaryOperator::BitwiseOr => OperatorType::Infix("|"),
                    BinaryOperator::BitwiseXor => OperatorType::Infix("^"),
                    BinaryOperator::BitwiseXnor => todo!("codegen BitwiseXnor"),
                };

                match op_equivalent {
                    OperatorType::Infix(name) => {
                        self.codegen_expr(file, *lhs, is_prev)?;
                        emit!(file, " {} ", name)?;
                        self.codegen_expr(file, *rhs, is_prev)?;
                    }

                    OperatorType::Function(name) => {
                        self.codegen_expr(file, *lhs, is_prev)?;
                        emit!(file, ".{}(", name)?;
                        self.codegen_expr(file, *rhs, is_prev)?;
                        emit!(file, ")")?;
                    }
                }
            }
            ExprKind::Concatenation { exprs: _ } => todo!("codegen Concatenation"),
            ExprKind::PartSelect { lhs, rhs, target } => {
                if is_prev {
                    emit!(file, "prev[tid].")?;
                } else {
                    emit!(file, "next[tid].")?;
                }
                self.codegen_signal_name(file, *target)?;
                emit!(file, ".select_part(")?;
                self.codegen_expr(file, *lhs, is_prev)?;
                emit!(file, ", ")?;
                self.codegen_expr(file, *rhs, is_prev)?;
                emit!(file, ")")?;
            }
            ExprKind::BitSelect { expr, target } => {
                if is_prev {
                    emit!(file, "prev[tid].")?;
                } else {
                    emit!(file, "next[tid].")?;
                }
                self.codegen_signal_name(file, *target)?;
                emit!(file, ".select_bit(")?;
                self.codegen_expr(file, *expr, is_prev)?;
                emit!(file, ")")?;
            }
            ExprKind::SignalRef { signal } => {
                if is_prev {
                    emit!(file, "prev[tid].")?;
                } else {
                    emit!(file, "next[tid].")?;
                }
                self.codegen_signal_name(file, *signal)?;
            }
            ExprKind::Invalid => unreachable!(),
        };

        Ok(())
    }

    fn codegen_signal_name<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        signal: SignalIdx,
    ) -> Result<()> {
        let signal = self.ast.get_signal(signal);
        emit!(file, "{}", clean_ident(&signal.full_name))?;

        Ok(())
    }

    fn codegen_signal_declaration<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        signal: SignalIdx,
    ) -> Result<()> {
        let ast_signal = self.ast.get_signal(signal);

        file.line_start()?;
        self.codegen_type(file, ast_signal.typ)?;
        emit!(file, " ")?;
        self.codegen_signal_name(file, signal)?;

        file.line_end_semicolon()?;

        Ok(())
    }

    fn codegen_type<W: Write>(&self, file: &mut CppEmitter<'a, W>, typ: TypeIdx) -> Result<()> {
        let typ = self.ast.get_typ(typ);

        match &typ.kind {
            TypeKind::Invalid => panic!("Invalid type propagated to codegen"),
            TypeKind::Integer | TypeKind::Int => {
                if typ.is_signed {
                    emit!(file, "int32_t")?;
                } else {
                    emit!(file, "uint32_t")?;
                }
            }
            TypeKind::Bit(range) | TypeKind::Logic(range) => {
                if let Some(range) = range {
                    emit!(file, "Bit<{}>", range.size())?;
                } else {
                    emit!(file, "Bit<1>")?;
                }
            }
        }

        Ok(())
    }
}

pub fn codegen_into_files<W: Write>(
    ast: &Ast,
    source: &mut W,
    header: &mut W,
    target: CodegenTarget,
) -> Result<()> {
    let mut source_emitter = CppEmitter::new(source);
    let mut header_emitter = CppEmitter::new(header);

    let codegen = Codegen::new(ast, target);

    codegen.codegen(&mut source_emitter, &mut header_emitter)?;

    Ok(())
}
