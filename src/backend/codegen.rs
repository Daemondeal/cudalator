use core::fmt;
use std::{collections::HashMap, io::Write};

use color_eyre::Result;

use crate::cir::{
    Ast, BinaryOperator, ConstantIdx, ConstantKind, ExprIdx, ExprKind, ModuleIdx, ProcessIdx,
    ScopeIdx, SelectKind, SensitivtyKind, SignalIdx, SignalLifetime, StatementIdx, StatementKind,
    TypeIdx, TypeKind, UnaryOperator,
};

pub enum CodegenTarget {
    CPU,
    CUDA,
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

fn process_name(process_idx: ProcessIdx) -> String {
    format!("process__{}", process_idx.get_idx())
}

struct Codegen<'a> {
    ast: &'a Ast,
    top_module_idx: ModuleIdx,
    target: CodegenTarget,

    top_name: String,

    top_signal_map: HashMap<SignalIdx, usize>,
}

impl<'a> Codegen<'a> {
    pub fn new(ast: &'a Ast, target: CodegenTarget) -> Self {
        let top_module_idx = ast.top_module.expect("No top module found");

        let top_module = ast.get_module(top_module_idx);
        let top_scope = ast.get_scope(top_module.scope);
        let top_name = clean_ident(&top_module.token.name);

        let mut top_signal_map = HashMap::new();
        let mut counter = 0;
        for signal in &top_scope.signals {
            top_signal_map.insert(*signal, counter);
            counter += 1;
        }

        Self {
            ast,
            top_module_idx,
            target,
            top_name,
            top_signal_map,
        }
    }

    pub fn codegen_function_qualifier<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>
    ) -> Result<()> {
        match self.target {
            CodegenTarget::CUDA => {
                emit!(file, "__global__ ")?;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn codegen<W: Write>(
        &self,
        source: &mut CppEmitter<'a, W>,
        header: &mut CppEmitter<'a, W>,
    ) -> Result<()> {
        let top_module = self.ast.get_module(self.top_module_idx);

        // Header
        emit!(header, "#pragma once\n")?;
        emit!(header, "#include \"../runtime/Process.hpp\"\n")?;
        emit!(header, "#include \"../runtime/Bit.hpp\"\n")?;
        emit!(header, "#include \"../runtime/ChangeType.hpp\"\n")?;
        emit!(header, "#include <vector>\n")?;
        emit!(header, "#include <cstddef>\n")?;
        emit!(header, "#include <fmt/os.h>\n")?;
        header.emit_empty_line()?;

        self.codegen_state_header(header)?;
        header.emit_empty_line()?;

        self.codegen_diff_header(header)?;
        header.emit_empty_line()?;

        self.codegen_process_container_header(header)?;
        header.emit_empty_line()?;

        self.codegen_top_struct_header(header)?;
        header.emit_empty_line()?;

        self.codegen_process_crossbar_header(header)?;
        header.emit_empty_line()?;

        // for process in &top_module.processes {
        //     self.codegen_process_header(header, *process)?;
        // }
        // header.emit_empty_line()?;

        self.codegen_vcd_dump_header(header)?;
        header.emit_empty_line()?;

        // Source
        emit!(source, "#include \"module.hpp\"\n")?;
        source.emit_empty_line()?;

        emit!(source, "#include \"../runtime/Vcd.hpp\"\n")?;
        source.emit_empty_line()?;

        self.codegen_diff_source(source)?;
        source.emit_empty_line()?;

        for process in &top_module.processes {
            self.codegen_process_body(source, *process)?;
            source.emit_empty_line()?;
        }

        self.codegen_process_container_source(source)?;
        source.emit_empty_line()?;

        self.codegen_process_crossbar_source(source)?;
        source.emit_empty_line()?;

        self.codegen_vcd_dump_source(source)?;
        source.emit_empty_line()?;

        Ok(())
    }

    fn codegen_top_struct_header<W: Write>(&self, header: &mut CppEmitter<'a, W>) -> Result<()> {
        // header.line_start()?;
        // emit!(header, "struct {}", self.top_name)?;
        // header.line_end()?;
        //
        // header.block_start()?;
        // header.block_end_semicolon()?;

        header.line_start()?;
        emit!(header, "using DiffType = diff_{}", self.top_name)?;
        header.line_end_semicolon()?;

        header.line_start()?;
        emit!(header, "using StateType = state_{}", self.top_name)?;
        header.line_end_semicolon()?;

        Ok(())
    }

    fn codegen_diff_source<W: Write>(&self, source: &mut CppEmitter<'a, W>) -> Result<()> {
        let top_module = self.ast.get_module(self.top_module_idx);
        let top_scope = self.ast.get_scope(top_module.scope);

        source.line_start()?;

        self.codegen_function_qualifier(source)?;
        emit!(
            source,
            "void state_calculate_diff(state_{}* start, state_{}* end, diff_{}* diffs)",
            self.top_name,
            self.top_name,
            self.top_name,
        )?;
        source.line_end()?;

        source.block_start()?;

        self.codegen_tid(source)?;


        for (i, signal) in top_scope.signals.iter().enumerate() {
            source.line_start()?;

            emit!(
                source,
                "diffs[tid].change[{i}] = change_calculate(start[tid].{}, end[tid].{})",
                self.signal_name(*signal),
                self.signal_name(*signal)
            )?;

            source.line_end_semicolon()?;
        }
        source.block_end()?;

        Ok(())
    }

    fn codegen_process_container_header<W: Write>(
        &self,
        header: &mut CppEmitter<'a, W>,
    ) -> Result<()> {
        header.line_start()?;
        emit!(
            header,
            "std::vector<Process<state_{}>> make_processes()",
            self.top_name
        )?;
        header.line_end_semicolon()?;

        Ok(())
    }

    fn codegen_process_container_source<W: Write>(
        &self,
        source: &mut CppEmitter<'a, W>,
    ) -> Result<()> {
        source.line_start()?;
        emit!(
            source,
            "std::vector<Process<state_{}>> make_processes()",
            self.top_name
        )?;
        source.line_end()?;

        source.block_start()?;

        source.line_start()?;
        emit!(
            source,
            "std::vector<Process<state_{}>> result",
            self.top_name
        )?;
        source.line_end_semicolon()?;

        let top_module = self.ast.get_module(self.top_module_idx);
        for process in &top_module.processes {
            let ast_process = self.ast.get_process(*process);

            source.line_start()?;
            emit!(source, "result.push_back(Process<state_{}>(", self.top_name)?;
            emit!(source, "\"{}\", ", process_name(*process))?;
            emit!(source, "{}, ", process.get_idx())?;

            // Emit the list of the ids of all signals inside the sensitivity list
            // TODO: Support negedge and posedge
            let signals = ast_process
                .sensitivity_list
                .iter()
                .map(|(kind, idx)| {
                    let sens = match kind {
                        SensitivtyKind::OnChange => "ChangeType::Change",
                        SensitivtyKind::Posedge => "ChangeType::Posedge",
                        SensitivtyKind::Negedge => "ChangeType::Negedge",
                    };

                    let signal_name = self
                        .top_signal_map
                        .get(idx)
                        .expect("signal not found in codegen")
                        .to_string();

                    format!("std::make_pair({signal_name}, {sens})")
                })
                .collect::<Vec<_>>()
                .join(", ");

            emit!(source, "{{{signals}}}))")?;
            source.line_end_semicolon()?;
        }

        source.line_start()?;
        emit!(source, "return result")?;
        source.line_end_semicolon()?;
        source.block_end()?;

        Ok(())
    }

    fn codegen_diff_header<W: Write>(&self, header: &mut CppEmitter<'a, W>) -> Result<()> {
        let top_module = self.ast.get_module(self.top_module_idx);
        let top_scope = self.ast.get_scope(top_module.scope);

        header.line_start()?;
        emit!(header, "struct diff_{}", self.top_name)?;
        header.line_end()?;

        header.block_start()?;

        header.line_start()?;
        emit!(header, "ChangeType change[{}]", top_scope.signals.len())?;
        header.line_end_semicolon()?;

        header.block_end_semicolon()?;
        header.emit_empty_line()?;

        header.line_start()?;
        self.codegen_function_qualifier(header)?;
        emit!(
            header,
            "void state_calculate_diff(state_{}* start, state_{}* end, diff_{}* diffs)",
            self.top_name,
            self.top_name,
            self.top_name,
        )?;
        header.line_end_semicolon()?;

        Ok(())
    }

    fn codegen_state_header<W: Write>(&self, header: &mut CppEmitter<'a, W>) -> Result<()> {
        // let top_module = self.ast.get_module(self.top_module_idx);
        // let top_scope = self.ast.get_scope(top_module.scope);

        header.line_start()?;
        emit!(header, "struct state_{}", self.top_name)?;
        header.line_end()?;

        header.block_start()?;
        for signal in gather_all_static_signals(self.ast) {
            self.codegen_signal_declaration(header, signal)?;
        }
        header.block_end_semicolon()?;

        Ok(())
    }

    fn codegen_process_crossbar_header<W: Write>(
        &self,
        header: &mut CppEmitter<'a, W>,
    ) -> Result<()> {

        header.line_start()?;
        self.codegen_function_qualifier(header)?;
        emit!(
            header,
            "void run_process(state_{} *state, size_t len, size_t process_idx)",
            self.top_name,
        )?;
        header.line_end_semicolon()?;
        Ok(())
    }

    fn codegen_process_crossbar_source<W: Write>(
        &self,
        source: &mut CppEmitter<'a, W>,
    ) -> Result<()> {
        let top_module = self.ast.get_module(self.top_module_idx);

        source.line_start()?;
        self.codegen_function_qualifier(source)?;
        emit!(
            source,
            "void run_process(state_{} *state, size_t len, size_t process_idx)",
            self.top_name,
        )?;
        source.line_end()?;

        source.block_start()?;

        source.line_start()?;
        emit!(source, "switch (process_idx)")?;
        source.line_end()?;

        source.block_start()?;
        for process in &top_module.processes {
            source.line_start()?;
            emit!(
                source,
                "case {}: {}(state, len); break",
                process.get_idx(),
                process_name(*process),
            )?;
            source.line_end_semicolon()?;
        }

        source.line_start()?;
        emit!(
            source,
            "default: break" // TODO: Implement error handling in GPU code
        )?;
        source.line_end_semicolon()?;
        source.block_end()?;

        source.block_end()?;

        Ok(())
    }

    fn codegen_process_header<W: Write>(
        &self,
        header: &mut CppEmitter<'a, W>,
        process: ProcessIdx,
    ) -> Result<()> {
        header.line_start()?;
        self.codegen_function_qualifier(header)?;
        emit!(
            header,
            "void {} (state_{} *state, size_t len)",
            process_name(process),
            self.top_name,
        )?;

        header.line_end_semicolon()?;
        Ok(())
    }

    fn codegen_process_body<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        process: ProcessIdx,
    ) -> Result<()> {
        let ast_process = self.ast.get_process(process);

        file.line_start()?;
        self.codegen_function_qualifier(file)?;
        emit!(
            file,
            "void {} (state_{} *state, size_t len)",
            process_name(process),
            self.top_name,
        )?;

        file.line_end()?;

        file.block_start()?;

        self.codegen_tid(file)?;

        let scope = self.ast.get_scope(ast_process.scope);
        for signal in &scope.signals {
            if self.ast.get_signal(*signal).lifetime == SignalLifetime::Automatic {
                self.codegen_signal_declaration(file, *signal)?;
            }
        }
        for statement in &ast_process.statements {
            self.codegen_statement(file, *statement)?;
        }

        file.block_end()?;

        Ok(())
    }

    fn codegen_scope<W: Write>(&self, file: &mut CppEmitter<'a, W>, scope: ScopeIdx) -> Result<()> {
        let scope = self.ast.get_scope(scope);

        for signal in &scope.signals {
            if self.ast.get_signal(*signal).lifetime == SignalLifetime::Automatic {
                self.codegen_signal_declaration(file, *signal)?;
            }
        }

        Ok(())
    }

    fn codegen_statement<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        statement: StatementIdx,
    ) -> Result<()> {
        let statement = self.ast.get_statement(statement);

        match &statement.kind {
            StatementKind::Assignment { lhs, rhs, select } => {
                file.line_start()?;

                self.codegen_signal_ref(file, *lhs)?;

                match select {
                    SelectKind::None => {
                        emit!(file, " = ")?;
                        self.codegen_expr(file, *rhs)?;
                    }
                    SelectKind::Bit(expr_idx) => {
                        emit!(file, ".set_bit(")?;
                        self.codegen_expr(file, *expr_idx)?;
                        emit!(file, ", ")?;
                        self.codegen_expr(file, *rhs)?;
                        emit!(file, ")")?;
                    }
                    SelectKind::Parts {
                        lhs: part_lhs,
                        rhs: part_rhs,
                    } => {
                        emit!(file, ".set_range(")?;
                        self.codegen_expr(file, *part_lhs)?;
                        emit!(file, ", ")?;
                        self.codegen_expr(file, *part_rhs)?;
                        emit!(file, ", ")?;
                        self.codegen_expr(file, *rhs)?;
                        emit!(file, ")")?;
                    }
                }

                file.line_end_semicolon()?;
            }

            StatementKind::Block { statements, scope } => {
                file.block_start()?;

                self.codegen_scope(file, *scope)?;

                for statement in statements {
                    self.codegen_statement(file, *statement)?;
                }

                file.block_end()?;
            }
            StatementKind::If {
                condition,
                body,
                else_,
            } => {
                file.line_start()?;
                emit!(file, "if (")?;
                self.codegen_expr(file, *condition)?;
                emit!(file, ")")?;
                file.line_end()?;

                self.codegen_statement(file, *body)?;

                if let Some(else_) = else_ {
                    file.line_start()?;
                    emit!(file, "else")?;
                    file.line_end()?;

                    self.codegen_statement(file, *else_)?;
                }
            }
            StatementKind::While { condition, body } => {
                file.line_start()?;
                emit!(file, "while (")?;
                self.codegen_expr(file, *condition)?;
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
                self.codegen_expr(file, *condition)?;
                emit!(file, ")")?;
                file.line_end_semicolon()?;
            }
            StatementKind::Case => todo!("codegen Case"),
            StatementKind::Foreach => todo!("codegen Foreach"),
            StatementKind::Return { expr } => {
                file.line_start()?;
                emit!(file, "return ")?;
                self.codegen_expr(file, *expr)?;
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

                emit!(file, "Bit<{}>({})", constant.size, vals[0])?;
            }
            // TODO: This should be done better
            ConstantKind::AllOnes => {
                todo!("codegen AllOnes")
                // emit!(file, "Bit<128>()")?;
            }
            ConstantKind::AllZero => {
                emit!(file, "Bit<128>(0)")?;
            }
            ConstantKind::Invalid => unreachable!(),
        };

        Ok(())
    }

    fn codegen_expr<W: Write>(&self, file: &mut CppEmitter<'a, W>, expr: ExprIdx) -> Result<()> {
        let expr = self.ast.get_expr(expr);

        match &expr.kind {
            ExprKind::Constant { constant } => self.codegen_constant(file, *constant)?,
            ExprKind::Unary { op, expr } => {
                enum UnaryOpType {
                    Prefix(&'static str),
                    Function(&'static str),
                }

                let op_equivalent = match op {
                    UnaryOperator::UnaryMinus => UnaryOpType::Prefix("-"),
                    UnaryOperator::UnaryPlus => UnaryOpType::Prefix("+"),
                    UnaryOperator::Not => UnaryOpType::Prefix("!"),
                    UnaryOperator::BinaryNegation => UnaryOpType::Prefix("~"),
                    UnaryOperator::ReductionAnd => UnaryOpType::Function("reduce_and"),
                    UnaryOperator::ReductionNand => UnaryOpType::Function("reduce_nand"),
                    UnaryOperator::ReductionOr => UnaryOpType::Function("reduce_or"),
                    UnaryOperator::ReductionNor => UnaryOpType::Function("reduce_nor"),
                    UnaryOperator::ReductionXor => UnaryOpType::Function("reduce_xor"),
                    UnaryOperator::ReductionXnor => UnaryOpType::Function("reduce_xnor"),
                    UnaryOperator::Posedge => unreachable!("should not codegen posedge"),
                    UnaryOperator::Negedge => unreachable!("should not codegen negedge"),
                };

                match op_equivalent {
                    UnaryOpType::Prefix(name) => {
                        emit!(file, "{}", name)?;
                        self.codegen_expr(file, *expr)?;
                    }

                    UnaryOpType::Function(name) => {
                        self.codegen_expr(file, *expr)?;
                        emit!(file, ".{}()", name)?;
                    }
                };
            }
            ExprKind::Binary { op, lhs, rhs } => {
                enum BinaryOpType {
                    Infix(&'static str),
                    Function(&'static str),
                }

                let op_equivalent = match op {
                    BinaryOperator::Subtraction => BinaryOpType::Infix("-"),
                    BinaryOperator::Division => BinaryOpType::Infix("/"),
                    BinaryOperator::Modulo => BinaryOpType::Infix("%"),
                    BinaryOperator::Equality => BinaryOpType::Infix("=="),
                    BinaryOperator::NotEquality => BinaryOpType::Infix("!="),
                    BinaryOperator::GreaterThan => BinaryOpType::Infix(">"),
                    BinaryOperator::GreaterThanEq => BinaryOpType::Infix(">="),
                    BinaryOperator::LessThan => BinaryOpType::Infix("<"),
                    BinaryOperator::LessThanEq => BinaryOpType::Infix("<="),
                    BinaryOperator::LeftShift => BinaryOpType::Infix("<<"),
                    BinaryOperator::RightShift => BinaryOpType::Infix(">>"),
                    BinaryOperator::Addition => BinaryOpType::Infix("+"),
                    BinaryOperator::Multiplication => BinaryOpType::Infix("*"),
                    BinaryOperator::Power => BinaryOpType::Function("pow"),
                    BinaryOperator::LogicalAnd => BinaryOpType::Infix("&&"),
                    BinaryOperator::LogicalOr => BinaryOpType::Infix("||"),
                    BinaryOperator::BitwiseAnd => BinaryOpType::Infix("&"),
                    BinaryOperator::BitwiseOr => BinaryOpType::Infix("|"),
                    BinaryOperator::BitwiseXor => BinaryOpType::Infix("^"),
                    BinaryOperator::BitwiseXnor => BinaryOpType::Function("xnor"),
                };

                match op_equivalent {
                    BinaryOpType::Infix(name) => {
                        emit!(file, "(")?;
                        self.codegen_expr(file, *lhs)?;
                        emit!(file, " {} ", name)?;
                        self.codegen_expr(file, *rhs)?;
                        emit!(file, ")")?;
                    }

                    BinaryOpType::Function(name) => {
                        self.codegen_expr(file, *lhs)?;
                        emit!(file, ".{}(", name)?;
                        self.codegen_expr(file, *rhs)?;
                        emit!(file, ")")?;
                    }
                }
            }
            ExprKind::Concatenation { exprs: _ } => todo!("codegen Concatenation"),
            ExprKind::PartSelect { lhs, rhs, target } => {
                self.codegen_signal_ref(file, *target)?;
                emit!(file, ".select_part(")?;
                self.codegen_expr(file, *lhs)?;
                emit!(file, ", ")?;
                self.codegen_expr(file, *rhs)?;
                emit!(file, ")")?;
            }
            ExprKind::BitSelect { expr, target } => {
                self.codegen_signal_ref(file, *target)?;
                emit!(file, ".select_bit(")?;
                self.codegen_expr(file, *expr)?;
                emit!(file, ")")?;
            }
            ExprKind::SignalRef { signal } => {
                self.codegen_signal_ref(file, *signal)?;
            }
            ExprKind::Invalid => unreachable!(),
        };

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
        emit!(file, " {}", self.signal_name(signal))?;

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

    fn codegen_tid<W: Write>(&self, source: &mut CppEmitter<'a, W>) -> Result<()> {
        source.line_start()?;
        match self.target {
            CodegenTarget::CPU => {
                emit!(source, "int tid = 0")?;
                source.line_end_semicolon()?;
            }
            CodegenTarget::CUDA => {
                emit!(source, "int tid = blockIdx.x * blockDim.x + threadIdx.x")?;
                source.line_end_semicolon()?;
                source.line_start()?;
                emit!(source, "if (tid > len) return")?;
                source.line_end_semicolon()?;
            }
        }

        Ok(())
    }

    fn signal_name(&self, idx: SignalIdx) -> String {
        let signal = self.ast.get_signal(idx);

        if signal.is_in_top_interface {
            signal.token.name.clone()
        } else {
            format!("{}_{}", clean_ident(&signal.full_name), idx.get_idx())
        }
    }

    fn signal_vcd_name(&self, idx: SignalIdx) -> &str {
        let signal = self.ast.get_signal(idx);

        signal.full_name.split_once('.')
          .map(|(_, rest)| rest)
          .unwrap_or(&signal.full_name)
    }

    // Generate the name to refer to a signal inside a kernel
    fn codegen_signal_ref<W: Write>(
        &self,
        file: &mut CppEmitter<'a, W>,
        idx: SignalIdx,
    ) -> Result<()> {
        let signal = self.ast.get_signal(idx);

        match signal.lifetime {
            // Automatic signals can simply be local variables
            SignalLifetime::Automatic => {
                emit!(file, "{}_{}", clean_ident(&signal.full_name), idx.get_idx())?;
            }

            SignalLifetime::Static | SignalLifetime::Net => {
                emit!(file, "state[tid].{}", self.signal_name(idx))?;
            }
        }

        Ok(())
    }

    fn codegen_vcd_dump_header<W: Write>(
        &self,
        header: &mut CppEmitter<'a, W>,
    ) -> Result<()> {
        header.line_start()?;
        emit!(header, "void state_vcd_dump_names(fmt::ostream &file)")?;
        header.line_end_semicolon()?;

        header.line_start()?;
        emit!(header, "void state_vcd_dump_values(state_{} *state, int tid, fmt::ostream &file)", self.top_name)?;
        header.line_end_semicolon()?;

        Ok(())
    }

    fn codegen_vcd_dump_source<W: Write>(
        &self,
        source: &mut CppEmitter<'a, W>,
    ) -> Result<()> {

        source.line_start()?;
        emit!(source, "void state_vcd_dump_names(fmt::ostream &file)")?;
        source.line_end()?;

        source.block_start()?;
        for idx in gather_all_static_signals(self.ast) {
            let signal = self.ast.get_signal(idx);
            source.line_start()?;
            emit!(
                source,
                "file.print(\"$var reg {} >{} {} $end\\n\")",
                signal.size(self.ast),
                idx.get_idx(),
                self.signal_vcd_name(idx)
            )?;
            source.line_end_semicolon()?;
        }
        source.block_end()?;

        source.line_start()?;
        emit!(source, "void state_vcd_dump_values(state_{} *state, int tid, fmt::ostream &file)", self.top_name)?;
        source.line_end()?;

        source.block_start()?;
        for idx in gather_all_static_signals(self.ast) {
            source.line_start()?;
            emit!(
                source,
                "file.print(\"b{{}} >{}\\n\", vcd_dump_value(",
                idx.get_idx(),
            )?;
            self.codegen_signal_ref(source, idx)?;
            emit!(source, "))")?;
            source.line_end_semicolon()?;
        }
        source.block_end()?;

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

fn gather_all_static_signals(ast: &Ast) -> Vec<SignalIdx> {
    // NOTE: Doing it this way also includes signals that may not be used anywhere. Probably sub
    //       this with an actual dfs on the ast tree.
    let mut result = vec![];

    for (i, signal) in ast.signals.iter().enumerate() {
        match signal.lifetime {
            SignalLifetime::Static | SignalLifetime::Net => {
                result.push(SignalIdx::from_idx(i as u32));
            }
            SignalLifetime::Automatic => {}
        }
    }

    result
}
