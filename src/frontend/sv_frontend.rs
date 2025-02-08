use core::fmt;
use log::warn;
use num_bigint::BigUint;
use num_traits::Num;
use surelog_sys::{VpiHandle, VpiValue};

use crate::cir::{
    Ast, BinaryOperator, Constant, ConstantIdx, ConstantKind, Expr, ExprIdx, ExprKind, Module,
    ModuleIdx, ModulePort, PortDirection, Process, ProcessIdx, Range, Scope, ScopeIdx,
    SensitivtyKind, Signal, SignalIdx, SignalLifetime, Statement, StatementIdx, StatementKind,
    Token, Type, TypeIdx, TypeKind, UnaryOperator,
};
use surelog_sys::bindings as sl;

use super::frontend_errors::FrontendError;

// Entry point for the frontend
pub fn compile_systemverilog(
    files: &[String],
    top_module: Option<&str>,
) -> Result<Ast, Vec<FrontendError>> {
    // FIXME: This is a bit ugly, maybe reconsider
    let refs = files.iter().map(|x| x.as_str()).collect::<Vec<_>>();

    let design = surelog_sys::compile(&refs, top_module);

    let top = design.get_top().expect("No top entity found");

    SvFrontend::translate_design(top)
}

// struct ProcessContext {
//     blocking_assign_alias: HashMap<SignalIdx>
// }

struct SvFrontend {
    ast: Ast,
    errors: Vec<FrontendError>,
    current_scope: Option<ScopeIdx>,
    temp_counter: usize,
}

// Utilities
fn token_from_vpi(vpi: VpiHandle) -> Token {
    let name = vpi.name();
    let line = vpi.vpi_get(sl::vpiLineNo);
    Token { name, line }
}

// TODO: Maybe actually implement these to remove the
//       dependency to num_bigint and num_traits here.
fn str_to_values_le(value: &str, radix: u32) -> Vec<u32> {
    // FIXME: Maybe we should propagate the error, idk if Surelog catches this already
    let big_uint = BigUint::from_str_radix(value, radix).expect("Invalid constant");

    big_uint.to_u32_digits()
}

fn evaluate_constant_to_u32(constant: &Constant) -> Result<u32, FrontendError> {
    match &constant.kind {
        ConstantKind::Integer(val) => {
            if *val < 0 {
                Err(FrontendError::other(
                    constant.token.clone(),
                    format!("Expected positive integer but got negative value"),
                ))
            } else if *val > (u32::max_value() as i64) {
                Err(FrontendError::other(
                    constant.token.clone(),
                    format!("Value {val} larger than {}", u32::max_value()),
                ))
            } else {
                Ok(*val as u32)
            }
        }
        ConstantKind::UnsignedInteger(val) => {
            if *val > (u32::max_value() as u64) {
                Err(FrontendError::other(
                    constant.token.clone(),
                    format!("Value {val} larger than {}", u32::max_value()),
                ))
            } else {
                Ok(*val as u32)
            }
        }
        ConstantKind::Value { vals } => {
            if vals.len() > 1 {
                Err(FrontendError::other(
                    constant.token.clone(),
                    format!("Value too big"),
                ))
            } else if vals.len() == 0 {
                unreachable!("constant has zero values")
            } else {
                Ok(vals[0])
            }
        }
        ConstantKind::Invalid => unreachable!(),
    }
}

impl SvFrontend {
    fn translate_design(top: VpiHandle) -> Result<Ast, Vec<FrontendError>> {
        let mut frontend = Self::new();
        let module = frontend.translate_module(top);
        frontend.ast.top_module = Some(module);

        if frontend.errors.is_empty() {
            Ok(frontend.ast)
        } else {
            Err(frontend.errors)
        }
    }

    fn new() -> Self {
        Self {
            ast: Ast::new(),
            errors: vec![],
            current_scope: None,
            temp_counter: 0,
        }
    }

    fn translate_module(&mut self, module: VpiHandle) -> ModuleIdx {
        assert!(module.vpi_type() == sl::vpiModule);

        let token = token_from_vpi(module);

        let scope_idx = self.ast.add_scope(Scope {
            token: token.clone(),
            parent: None,
            signals: vec![],
            is_top: true,
        });

        self.push_scope(scope_idx);

        let mut ast_module = Module {
            token,
            scope: scope_idx,
            ports: vec![],
            processes: vec![],
        };

        self.translate_module_flattened(module, &mut ast_module, true);

        self.ast.add_module(ast_module)
    }

    fn translate_module_flattened(
        &mut self,
        module: VpiHandle,
        ast_module: &mut Module,
        is_top: bool,
    ) {
        for net in module.vpi_iter(sl::vpiNet) {
            self.translate_net(net);
        }

        for var in module.vpi_iter(sl::vpiVariables) {
            self.translate_variable(var);
        }

        for port in module.vpi_iter(sl::vpiPort) {
            if is_top {
                match self.translate_port_top(port) {
                    Ok(port_idx) => ast_module.ports.push(port_idx),
                    Err(err) => self.errors.push(err),
                };
            } else {
                // Ports in submodules get automatically connected to the corresponding
                // signal in the top module
                if let Some(ast_connection) = self.translate_port_sub(port) {
                    ast_module.processes.push(ast_connection);
                }
            }
        }

        for cont_assign in module.vpi_iter(sl::vpiContAssign) {
            let proc_idx = self.translate_continuous_assignment(cont_assign);
            ast_module.processes.push(proc_idx);
        }

        for process in module.vpi_iter(sl::vpiProcess) {
            let token = token_from_vpi(process);
            let proc_type = process.vpi_type();

            match proc_type {
                sl::vpiInitial => warn!(
                    "Only always processes are implemented, skipping initial process at line {}...",
                    token.line
                ),
                sl::vpiFinal => warn!(
                    "Only always processes are implemented, skipping final process at line {}...",
                    token.line
                ),
                sl::vpiAlways => {
                    let proc_idx = self.translate_always(process);
                    ast_module.processes.push(proc_idx);
                }

                other => panic!("Invalid process type {other}"),
            }
        }

        for submodule in module.vpi_iter(sl::vpiModule) {
            self.translate_module_flattened(submodule, ast_module, false);
        }

        // TODO: Add gen scope array
    }

    fn translate_port_top(&mut self, port: VpiHandle) -> Result<ModulePort, FrontendError> {
        assert!(port.vpi_type() == sl::vpiPort);
        let token = token_from_vpi(port);

        let direction = match port.vpi_get(sl::vpiDirection) as u32 {
            sl::vpiInput => PortDirection::Input,
            sl::vpiOutput => PortDirection::Output,
            sl::vpiInout => PortDirection::Inout,
            _ => {
                self.errors.push(FrontendError::other(
                    token.clone(),
                    "Invalid port direction".to_owned(),
                ));
                PortDirection::Invalid
            }
        };

        let low_conn = port.vpi_handle(sl::vpiLowConn).expect("No low conn found");

        // TODO: Add a proper error here
        let signal_idx = self.get_signal_from_ref(low_conn).ok_or_else(|| {
            FrontendError::other(token.clone(), format!("Cannot find signal {}", token.name))
        })?;

        Ok(ModulePort {
            signal: signal_idx,
            direction,
        })
    }

    fn translate_port_sub(&mut self, port: VpiHandle) -> Option<ProcessIdx> {
        let token = token_from_vpi(port);

        let direction = match port.vpi_get(sl::vpiDirection) as u32 {
            sl::vpiInput => PortDirection::Input,
            sl::vpiOutput => PortDirection::Output,
            sl::vpiInout => {
                self.err_other(&token, format_args!("inout port"));
                return None;
            }
            _ => {
                self.err_other(&token, format_args!("Invalid port direction"));
                return None;
            }
        };

        let low_conn = port.vpi_handle(sl::vpiLowConn).expect("No low conn found");
        let Some(high_conn) = port.vpi_handle(sl::vpiHighConn) else {
            warn!(
                "Line {}: No high conn fonud for port {}",
                token.line, token.name
            );
            return None;
        };

        let low = self.translate_expr(low_conn);
        let high = self.translate_expr(high_conn);

        let (lhs, rhs) = match direction {
            PortDirection::Input => (low, high),
            PortDirection::Output => (high, low),
            _ => unreachable!(),
        };

        let assignment = self.ast.add_statement(Statement {
            token: token.clone(),
            kind: StatementKind::Assignment {
                lhs,
                rhs,
                blocking: false,
            },
        });

        Some(self.ast.add_process(Process {
            token,
            statement: assignment,
            sensitivity_list: vec![],
            should_populate_sensitivity_list: true,
        }))
    }

    fn translate_net(&mut self, net: VpiHandle) -> SignalIdx {
        assert!(net.vpi_type() == sl::vpiNet);
        let token = token_from_vpi(net);

        let full_name = net.vpi_str(sl::vpiFullName);

        let type_spec = net.vpi_handle(sl::vpiTypespec).expect("No type spec found");
        let type_idx = self.translate_typespec(type_spec);

        let signal_idx = self.ast.add_signal(Signal {
            token,
            full_name,
            typ: type_idx,
            lifetime: SignalLifetime::Net,
            scope: self
                .current_scope
                .expect("Initalizating signal out of scoe"),
        });

        self.add_signal_to_current_scope(signal_idx);
        signal_idx
    }

    fn translate_variable(&mut self, var: VpiHandle) -> SignalIdx {
        let token = token_from_vpi(var);

        // TODO: Variables have more parameters than this, actually check for them.
        let full_name = var.vpi_str(sl::vpiFullName);

        let lifetime = if var.vpi_get_bool(sl::vpiAutomatic) {
            SignalLifetime::Automatic
        } else {
            SignalLifetime::Static
        };

        let type_spec = var.vpi_handle(sl::vpiTypespec).expect("No type spec found");
        let type_idx = self.translate_typespec(type_spec);

        let signal_idx = self.ast.add_signal(Signal {
            token,
            full_name,
            typ: type_idx,
            lifetime,
            scope: self.current_scope.expect("Signal declared outside a scope"),
        });

        self.add_signal_to_current_scope(signal_idx);
        signal_idx
    }

    fn translate_always(&mut self, always: VpiHandle) -> ProcessIdx {
        let token = token_from_vpi(always);

        let is_always_comb = always.vpi_get(sl::vpiAlwaysType) == sl::vpiAlwaysComb;

        let mut stmt = always.vpi_handle(sl::vpiStmt).expect("Process has no body");

        let mut sensitivity_list = vec![];
        if stmt.vpi_type() == sl::vpiEventControl {
            let condition = stmt
                .vpi_handle(sl::vpiCondition)
                .expect("Event control has no condition");

            self.translate_sensitivity_list(&mut sensitivity_list, condition);

            // TODO: Maybe we should allow empty processes
            stmt = stmt.vpi_handle(sl::vpiStmt).expect("Empty process");
        };

        let stmt_idx = self.translate_statement(stmt);

        self.ast.add_process(Process {
            token,
            statement: stmt_idx,
            sensitivity_list,
            should_populate_sensitivity_list: is_always_comb,
        })
    }

    fn translate_sensitivity_list(
        &mut self,
        sensitivity: &mut Vec<(SensitivtyKind, SignalIdx)>,
        condition: VpiHandle,
    ) {
        match condition.vpi_type() {
            sl::vpiOperation => {
                let op_type = condition.vpi_get(sl::vpiOpType);

                match op_type {
                    sl::vpiListOp => {
                        for operand in condition.vpi_iter(sl::vpiOperand) {
                            self.translate_sensitivity_list(sensitivity, operand);
                        }
                    }
                    sl::vpiNegedgeOp | sl::vpiPosedgeOp => {
                        let kind = if op_type == sl::vpiNegedgeOp {
                            SensitivtyKind::Negedge
                        } else {
                            SensitivtyKind::Posedge
                        };

                        let first_op = condition
                            .vpi_iter(sl::vpiOperand)
                            .next()
                            .expect("Sensitivity list element has no operand");

                        assert!(first_op.vpi_type() == sl::vpiRefObj);
                        let Some(signal) = self.get_signal_from_ref(first_op) else {
                            self.err_signal_not_found(first_op);
                            return;
                        };

                        sensitivity.push((kind, signal));
                    }
                    _ => {
                        panic!("Sensitivity list element was neither an op nor a ref_obj");
                    }
                }
            }
            sl::vpiRefObj => {
                let Some(signal) = self.get_signal_from_ref(condition) else {
                    self.err_signal_not_found(condition);
                    return;
                };

                sensitivity.push((SensitivtyKind::OnChange, signal))
            }
            _ => panic!("Invalid condition type for event control"),
        }
    }

    /*
    fn translate_assignment(&mut self, assignment: VpiHandle) -> StatementIdx {
        assert!(assignment.vpi_type() == sl::vpiAssignment);
        let token = token_from_vpi(assignment);

        let rhs = self.translate_expr(assignment.vpi_handle(sl::vpiRhs).expect("No rhs found"));

        let vpi_lhs = assignment.vpi_handle(sl::vpiLhs).expect("No lhs found");

        let lhs_kind = match vpi_lhs.vpi_type() {
            sl::vpiConcatOp => todo!("translate_assignment vpiConcatOp"),

            sl::vpiPartSelect => todo!(),
            sl::vpiBitSelect => todo!(),
            sl::vpiRefObj | sl::vpiRefVar => {

            }

            sl::vpiLongIntVar
            | sl::vpiShortIntVar
            | sl::vpiIntVar
            | sl::vpiShortRealVar
            | sl::vpiByteVar
            | sl::vpiClassVar
            | sl::vpiStringVar
            | sl::vpiEnumVar
            | sl::vpiStructVar
            | sl::vpiUnionVar
            | sl::vpiBitVar
            | sl::vpiReg
            | sl::vpiRegArray
            | sl::vpiClassObj
            | sl::vpiChandleVar
            | sl::vpiPackedArrayVar
            | sl::vpiVirtualInterfaceVar => todo!(),

            _ => {
                self.err_other(
                    &token,
                    format_args!("Unsupported assignment {}", assignment.vpi_type()),
                );
            }
        };

        todo!()
    }
    */

    fn translate_statement(&mut self, statement: VpiHandle) -> StatementIdx {
        let token = token_from_vpi(statement);
        let stmt_type = statement.vpi_type();

        let kind = match stmt_type {
            sl::vpiBegin | sl::vpiNamedBegin => {
                let ast_scope = self.ast.add_scope(Scope {
                    token: token.clone(),
                    parent: self.current_scope,
                    signals: vec![],
                    is_top: false,
                });

                self.push_scope(ast_scope);

                for signal in statement.vpi_iter(sl::vpiVariables) {
                    self.translate_variable(signal);
                }

                let statements = statement
                    .vpi_iter(sl::vpiStmt)
                    .map(|s| self.translate_statement(s))
                    .collect::<Vec<_>>();

                self.pop_scope();

                StatementKind::Block {
                    statements,
                    scope: ast_scope,
                }
            }

            sl::vpiFor => {
                let ast_scope = self.ast.add_scope(Scope {
                    token: token.clone(),
                    parent: self.current_scope,
                    signals: vec![],
                    is_top: false,
                });

                self.push_scope(ast_scope);

                for signal in statement.vpi_iter(sl::vpiVariables) {
                    self.translate_variable(signal);
                }

                // For statements will get translated into:
                // {
                //   (initializers)
                //   while (condition) {
                //     (body)
                //     (incrementer)
                //   }
                // }

                let mut outer_statements = statement
                    .vpi_iter(sl::vpiForInitStmt)
                    .map(|s| self.translate_statement(s))
                    .collect::<Vec<_>>();

                // let init_stmt = if inits.len() == 1 {
                //     inits[0]
                // } else {
                //     self.ast.add_statement(Statement {
                //         token: token.clone(),
                //         kind: StatementKind::Block { statements: inits },
                //     })
                // };

                let condition = match statement.vpi_handle(sl::vpiCondition) {
                    Some(cond) => self.translate_expr(cond),
                    None => self.make_constant_true(),
                };

                let mut incrs = statement
                    .vpi_iter(sl::vpiForIncStmt)
                    .map(|s| self.translate_statement(s))
                    .collect::<Vec<_>>();

                // let incr_stmt = if incrs.len() == 1 {
                //     incrs[0]
                // } else {
                //     self.ast.add_statement(Statement {
                //         token: token.clone(),
                //         kind: StatementKind::Block { statements: incrs },
                //     })
                // };

                let body = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiStmt)
                        .expect("No for body found"),
                );

                let ast_body = self.ast.get_statement_mut(body);

                let mut was_block = false;
                match &mut ast_body.kind {
                    StatementKind::Block {
                        statements,
                        scope: _,
                    } => {
                        statements.append(&mut incrs);
                        was_block = true;
                    }
                    _ => {}
                }

                // If the inside body was not a block, we need to create one ourselves to
                // add the increment statements at the end
                let body = if was_block {
                    body
                } else {
                    incrs.insert(0, body);
                    let scope = self.ast.add_scope(Scope {
                        is_top: false,
                        token: token.clone(),
                        parent: self.current_scope,
                        signals: vec![],
                    });

                    self.ast.add_statement(Statement {
                        token: token.clone(),
                        kind: StatementKind::Block {
                            statements: incrs,
                            scope,
                        },
                    })
                };

                self.pop_scope();

                let while_ = self.ast.add_statement(Statement {
                    token: token.clone(),
                    kind: StatementKind::While { condition, body },
                });

                outer_statements.push(while_);

                // let outer_block = self.ast.add_statement(statement)?;

                StatementKind::Block {
                    statements: outer_statements,
                    scope: ast_scope,
                }
                // StatementKind::For {
                //     condition,
                //     init: init_stmt,
                //     increment: incr_stmt,
                //     body,
                //     scope: ast_scope,
                // }
            }

            sl::vpiIf => {
                let condition = self.translate_expr(
                    statement
                        .vpi_handle(sl::vpiCondition)
                        .expect("No if condition found"),
                );
                let body = self.translate_statement(
                    statement.vpi_handle(sl::vpiStmt).expect("No if body found"),
                );

                StatementKind::If {
                    condition,
                    body,
                    else_: None,
                }
            }
            sl::vpiIfElse => {
                let condition = self.translate_expr(
                    statement
                        .vpi_handle(sl::vpiCondition)
                        .expect("No if_else condition found"),
                );
                let body = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiStmt)
                        .expect("No if_else body found"),
                );
                let else_ = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiElseStmt)
                        .expect("No if_else else found"),
                );

                StatementKind::If {
                    condition,
                    body,
                    else_: Some(else_),
                }
            }
            sl::vpiWhile => {
                let condition = self.translate_expr(
                    statement
                        .vpi_handle(sl::vpiCondition)
                        .expect("No while condition found"),
                );
                let body = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiStmt)
                        .expect("No while body found"),
                );

                StatementKind::While { condition, body }
            }
            sl::vpiDoWhile => {
                let condition = self.translate_expr(
                    statement
                        .vpi_handle(sl::vpiCondition)
                        .expect("No do_while condition found"),
                );
                let body = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiStmt)
                        .expect("No do_while body found"),
                );

                StatementKind::DoWhile { condition, body }
            }
            sl::vpiRepeat => {
                let condition = self.translate_expr(
                    statement
                        .vpi_handle(sl::vpiCondition)
                        .expect("No repeat condition found"),
                );
                let body = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiStmt)
                        .expect("No repeat body found"),
                );

                // This will get translated as:
                // {
                //   tmp_max = (condition)
                //   tmp_idx = 0
                //   while (tmp_idx < tmp_max) {
                //     body
                //     tmp_idx += 1
                //   }
                // }

                // We'll put these temporaries in the top scope, hopefully this is not a
                // problme.
                let int_typ = self.make_int_typ();
                let idx = self.make_temp_signal(int_typ);
                let max_count = self.make_temp_signal(int_typ);

                let idx_ref = self.ast.add_expr(Expr {
                    token: token.clone(),
                    kind: ExprKind::SignalRef { signal: idx },
                });
                let max_ref = self.ast.add_expr(Expr {
                    token: token.clone(),
                    kind: ExprKind::SignalRef { signal: max_count },
                });
                let zero = self.make_constant_zero();
                let one = self.make_constant_true();

                let setup_max = self.ast.add_statement(Statement {
                    token: token.clone(),
                    kind: StatementKind::Assignment {
                        lhs: max_ref,
                        rhs: condition,
                        blocking: true,
                    },
                });
                let setup_idx = self.ast.add_statement(Statement {
                    token: token.clone(),
                    kind: StatementKind::Assignment {
                        lhs: idx_ref,
                        rhs: zero,
                        blocking: true,
                    },
                });

                let stop_condition = self.ast.add_expr(Expr {
                    token: token.clone(),
                    kind: ExprKind::Binary {
                        op: BinaryOperator::LessThan,
                        lhs: idx_ref,
                        rhs: max_ref,
                    },
                });

                let incr_expr = self.ast.add_expr(Expr {
                    token: token.clone(),
                    kind: ExprKind::Binary {
                        op: BinaryOperator::Addition,
                        lhs: idx_ref,
                        rhs: one,
                    },
                });

                let incr_statement = self.ast.add_statement(Statement {
                    token: token.clone(),
                    kind: StatementKind::Assignment {
                        lhs: idx_ref,
                        rhs: incr_expr,
                        blocking: true,
                    },
                });

                let body = self.add_to_block_end(body, incr_statement);

                let while_loop = self.ast.add_statement(Statement {
                    token: token.clone(),
                    kind: StatementKind::While {
                        condition: stop_condition,
                        body,
                    },
                });

                let statements = vec![setup_max, setup_idx, while_loop];

                StatementKind::Block {
                    statements,
                    scope: self.current_scope.expect("No scope found"),
                }
            }
            sl::vpiAssignment => {
                let blocking = statement.vpi_get_bool(sl::vpiBlocking);
                let lhs =
                    self.translate_expr(statement.vpi_handle(sl::vpiLhs).expect("No lhs found"));
                let rhs =
                    self.translate_expr(statement.vpi_handle(sl::vpiRhs).expect("No rhs found"));

                let op = statement.vpi_get(sl::vpiOpType);

                // FIXME: Standard 11.4.1, any left-hand index operation is only supposed to
                //        be evaluated once, but we don't really have a way to enforce this yet.
                //        Look into this eventually.
                // NOTE : Sometimes the normal assignment returns zero as the operation type.
                //        This is not what the standard says but oh well.
                let actual_rhs = if op == sl::vpiAssignmentOp || op == 0 {
                    rhs
                } else {
                    let op_kind = match op {
                        sl::vpiAddOp => Some(BinaryOperator::Addition),
                        sl::vpiSubOp => Some(BinaryOperator::Subtraction),
                        sl::vpiMultOp => Some(BinaryOperator::Multiplication),
                        sl::vpiDivOp => Some(BinaryOperator::Division),
                        sl::vpiModOp => Some(BinaryOperator::Modulo),
                        sl::vpiPowerOp => Some(BinaryOperator::Power),
                        sl::vpiBitAndOp => Some(BinaryOperator::BitwiseAnd),
                        sl::vpiBitOrOp => Some(BinaryOperator::BitwiseOr),
                        sl::vpiBitXorOp => Some(BinaryOperator::BitwiseXor),
                        sl::vpiRShiftOp => Some(BinaryOperator::RightShift),
                        sl::vpiLShiftOp => Some(BinaryOperator::LeftShift),
                        sl::vpiArithLShiftOp => todo!("arithmetic shifts"),
                        sl::vpiArithRShiftOp => todo!("arithmetic shifts"),
                        _ => None,
                    };

                    if let Some(kind) = op_kind {
                        self.ast.add_expr(Expr {
                            token: token.clone(),
                            kind: ExprKind::Binary { op: kind, lhs, rhs },
                        })
                    } else {
                        self.err_unsupported(&token, format_args!("assignment operation {op}"));
                        rhs
                    }
                };

                StatementKind::Assignment {
                    lhs,
                    rhs: actual_rhs,
                    blocking,
                }
            }
            sl::vpiForever => {
                let body = self.translate_statement(
                    statement
                        .vpi_handle(sl::vpiStmt)
                        .expect("No forever body found"),
                );

                StatementKind::While {
                    condition: self.make_constant_true(),
                    body,
                }
            }

            sl::vpiReturnStmt => {
                let expr = self.translate_expr(
                    statement
                        .vpi_handle(sl::vpiCondition)
                        .expect("No forever body found"),
                );

                StatementKind::Return { expr }
            }

            // No parameters
            sl::vpiBreak => StatementKind::Break,
            sl::vpiContinue => StatementKind::Continue,
            sl::vpiNullStmt => {
                // FIXME: Remove this when project is done
                warn!("Line {} null statement found.", token.line);
                StatementKind::Null
            }

            // TODO: Implement these
            sl::vpiForeachStmt => {
                todo!("vpiForeachStmt")
            }
            sl::vpiCase => {
                todo!("vpiCase")
            }

            // Unsupported statements
            sl::vpiEventStmt => {
                self.err_unsupported(&token, format_args!("statement \"event statement\""));
                StatementKind::Invalid
            }
            sl::vpiForce => {
                self.err_unsupported(&token, format_args!("statement \"force\""));
                StatementKind::Invalid
            }
            sl::vpiRelease => {
                self.err_unsupported(&token, format_args!("statement \"release\""));
                StatementKind::Invalid
            }
            sl::vpiDeassign => {
                self.err_unsupported(&token, format_args!("statement \"deassign\""));
                StatementKind::Invalid
            }
            sl::vpiAssignStmt => {
                self.err_unsupported(&token, format_args!("statement \"assign\""));
                StatementKind::Invalid
            }
            sl::vpiDelayControl => {
                self.err_unsupported(&token, format_args!("statement \"delay control\""));
                StatementKind::Invalid
            }
            sl::vpiEventControl => {
                self.err_unsupported(&token, format_args!("statement \"event control\""));
                StatementKind::Invalid
            }
            sl::vpiExpectStmt => {
                self.err_unsupported(&token, format_args!("statement \"expect\""));
                StatementKind::Invalid
            }
            sl::vpiImmediateAssert => {
                self.err_unsupported(&token, format_args!("statement \"immediate assert\""));
                StatementKind::Invalid
            }
            sl::vpiImmediateAssume => {
                self.err_unsupported(&token, format_args!("statement \"immediate assume\""));
                StatementKind::Invalid
            }
            sl::vpiImmediateCover => {
                self.err_unsupported(&token, format_args!("statement \"immediate cover\""));
                StatementKind::Invalid
            }

            sl::vpiFork => {
                self.err_unsupported(&token, format_args!("statement \"fork\""));
                StatementKind::Invalid
            }

            sl::vpiNamedFork => {
                self.err_unsupported(&token, format_args!("statement \"fork\""));
                StatementKind::Invalid
            }

            // TODO: Figure out what these are
            // sl::vpiWaits => {},
            // sl::vpiDisables => {},
            // sl::vpiTfCall => {},
            _ => {
                self.err_unsupported(&token, format_args!("statement type {stmt_type}"));
                StatementKind::Invalid
            }
        };

        self.ast.add_statement(Statement { token, kind })
    }

    fn translate_continuous_assignment(&mut self, cont_assign: VpiHandle) -> ProcessIdx {
        assert!(cont_assign.vpi_type() == sl::vpiContAssign);
        let token = token_from_vpi(cont_assign);

        let lhs = cont_assign
            .vpi_handle(sl::vpiLhs)
            .expect("No lhs found for cont assign");
        let rhs = cont_assign
            .vpi_handle(sl::vpiRhs)
            .expect("No rhs found for cont assign");

        let ast_lhs = self.translate_expr(lhs);
        let ast_rhs = self.translate_expr(rhs);

        let assignment = Statement {
            token: token.clone(),
            kind: StatementKind::Assignment {
                lhs: ast_lhs,
                rhs: ast_rhs,
                blocking: false,
            },
        };

        let assignment_idx = self.ast.add_statement(assignment);

        self.ast.add_process(Process {
            token,
            statement: assignment_idx,
            sensitivity_list: vec![],
            should_populate_sensitivity_list: true,
        })
    }

    fn translate_expr(&mut self, expr: VpiHandle) -> ExprIdx {
        let token = token_from_vpi(expr);

        let kind = match expr.vpi_type() {
            sl::vpiConstant => {
                let constant = self.translate_constant(expr);
                ExprKind::Constant { constant }
            }
            sl::vpiOperation => {
                let op_type = expr.vpi_get(sl::vpiOpType);

                let unary = match op_type {
                    sl::vpiMinusOp => Some(UnaryOperator::UnaryMinus),
                    sl::vpiPlusOp => Some(UnaryOperator::UnaryPlus),
                    sl::vpiNotOp => Some(UnaryOperator::Not),
                    sl::vpiBitNegOp => Some(UnaryOperator::BinaryNegation),
                    sl::vpiUnaryAndOp => Some(UnaryOperator::ReductionAnd),
                    sl::vpiUnaryNandOp => Some(UnaryOperator::ReductionNand),
                    sl::vpiUnaryOrOp => Some(UnaryOperator::ReductionOr),
                    sl::vpiUnaryNorOp => Some(UnaryOperator::ReductionNor),
                    sl::vpiUnaryXorOp => Some(UnaryOperator::ReductionXor),
                    sl::vpiUnaryXNorOp => Some(UnaryOperator::ReductionXnor),
                    sl::vpiPosedgeOp => Some(UnaryOperator::Posedge),
                    sl::vpiNegedgeOp => Some(UnaryOperator::Negedge),
                    _ => None,
                };

                if let Some(unary_kind) = unary {
                    let operand = expr
                        .vpi_iter(sl::vpiOperand)
                        .next()
                        .expect("Cannot find operand for expression");

                    ExprKind::Unary {
                        op: unary_kind,
                        expr: self.translate_expr(operand),
                    }
                } else {
                    let binary = match op_type {
                        sl::vpiSubOp => Some(BinaryOperator::Subtraction),
                        sl::vpiDivOp => Some(BinaryOperator::Division),
                        sl::vpiModOp => Some(BinaryOperator::Modulo),
                        sl::vpiEqOp => Some(BinaryOperator::Equality),
                        sl::vpiNeqOp => Some(BinaryOperator::NotEquality),
                        sl::vpiGtOp => Some(BinaryOperator::GreaterThan),
                        sl::vpiGeOp => Some(BinaryOperator::GreaterThanEq),
                        sl::vpiLtOp => Some(BinaryOperator::LessThan),
                        sl::vpiLeOp => Some(BinaryOperator::LessThanEq),
                        sl::vpiLShiftOp => Some(BinaryOperator::LeftShift),
                        sl::vpiRShiftOp => Some(BinaryOperator::RightShift),
                        sl::vpiAddOp => Some(BinaryOperator::Addition),
                        sl::vpiMultOp => Some(BinaryOperator::Multiplication),
                        sl::vpiLogAndOp => Some(BinaryOperator::LogicalAnd),
                        sl::vpiLogOrOp => Some(BinaryOperator::LogicalOr),
                        sl::vpiBitAndOp => Some(BinaryOperator::BitwiseAnd),
                        sl::vpiBitOrOp => Some(BinaryOperator::BitwiseOr),
                        sl::vpiBitXorOp => Some(BinaryOperator::BitwiseXor),
                        sl::vpiBitXnorOp => Some(BinaryOperator::BitwiseXnor),

                        sl::vpiCaseEqOp => todo!(),
                        sl::vpiCaseNeqOp => todo!(),

                        _ => None,
                    };

                    if let Some(binary_kind) = binary {
                        let mut operands = expr.vpi_iter(sl::vpiOperand);

                        let lhs = self.translate_expr(operands.next().expect("Cannot find lhs"));
                        let rhs = self.translate_expr(operands.next().expect("Cannot find Rhs"));

                        ExprKind::Binary {
                            op: binary_kind,
                            lhs,
                            rhs,
                        }
                    } else if op_type == sl::vpiConcatOp {
                        let operands = expr
                            .vpi_iter(sl::vpiOperand)
                            .map(|opr| self.translate_expr(opr))
                            .collect::<Vec<_>>();

                        ExprKind::Concatenation { exprs: operands }
                    } else {
                        self.errors.push(FrontendError::other(
                            token.clone(),
                            format!("Unsupported operand type {op_type}"),
                        ));

                        ExprKind::Invalid
                    }
                }
            }
            sl::vpiPartSelect => {
                let left_range = expr
                    .vpi_handle(sl::vpiLeftRange)
                    .expect("No left range found");
                let right_range = expr
                    .vpi_handle(sl::vpiRightRange)
                    .expect("No right range found");

                let lhs = self.translate_expr(left_range);
                let rhs = self.translate_expr(right_range);

                if let Some(signal) = self.get_signal_from_ref(expr) {
                    ExprKind::PartSelect {
                        lhs,
                        rhs,
                        target: signal,
                    }
                } else {
                    self.err_signal_not_found(expr);
                    ExprKind::Invalid
                }
            }
            sl::vpiBitSelect => {
                let index = expr.vpi_handle(sl::vpiIndex).expect("No index found");

                let index_idx = self.translate_expr(index);

                if let Some(signal) = self.get_signal_from_ref(expr) {
                    ExprKind::BitSelect {
                        expr: index_idx,
                        target: signal,
                    }
                } else {
                    self.err_signal_not_found(expr);
                    ExprKind::Invalid
                }
            }
            sl::vpiRefObj | sl::vpiRefVar => {
                if let Some(signal) = self.get_signal_from_ref(expr) {
                    ExprKind::SignalRef { signal }
                } else {
                    self.err_signal_not_found(expr);
                    ExprKind::Invalid
                }
            }

            // All variable types
            sl::vpiLongIntVar
            | sl::vpiShortIntVar
            | sl::vpiIntVar
            | sl::vpiShortRealVar
            | sl::vpiByteVar
            | sl::vpiClassVar
            | sl::vpiStringVar
            | sl::vpiEnumVar
            | sl::vpiStructVar
            | sl::vpiUnionVar
            | sl::vpiBitVar
            | sl::vpiReg
            | sl::vpiRegArray
            | sl::vpiClassObj
            | sl::vpiChandleVar
            | sl::vpiPackedArrayVar
            | sl::vpiVirtualInterfaceVar => {
                let full_name = expr.vpi_str(sl::vpiFullName);
                // TODO: I can't find where this is specified, but it seems like if a signal is
                //       referred directly rather than through a reference, the expression where it
                //       is found is the initializer. Implement a way to detect this.

                let scope = self
                    .ast
                    .get_scope(self.current_scope.expect("Variable outside scope"));
                let var = scope.find_signal_recursively(&self.ast, &full_name);

                let var_idx = match var {
                    Some(var_idx) => var_idx,

                    // So, as it turns out sometimes Surelog decides to declare signals
                    // inside scopes in an expression instead of the vpiVariables field like
                    // it always does, like for example inside for statement initializers.
                    // This is a bit hacky, but it should be enough to handle that weird
                    // behavior.
                    None => {
                        let signal_idx = self.translate_variable(expr);

                        let scope = self
                            .ast
                            .get_scope_mut(self.current_scope.expect("Variable outside scope"));
                        scope.signals.push(signal_idx);
                        signal_idx
                    }
                };

                ExprKind::SignalRef { signal: var_idx }
            }

            _ => {
                self.errors.push(FrontendError::unsupported(
                    token.clone(),
                    format!("Unimplemented expr {}", expr.vpi_type()),
                ));
                ExprKind::Invalid
            }
        };

        self.ast.add_expr(Expr { token, kind })
    }

    fn translate_typespec(&mut self, typespec_ref: VpiHandle) -> TypeIdx {
        let token = token_from_vpi(typespec_ref);

        let typespec = typespec_ref
            .vpi_handle(sl::vpiActual)
            .expect("No actual typespec found");
        let typ = typespec.vpi_type();

        match typ {
            sl::vpiIntTypespec => {
                let is_signed = typespec.vpi_get(sl::vpiSigned) == 1;

                self.ast.add_typ(Type {
                    token,
                    kind: TypeKind::Int,
                    is_signed,
                })
            }
            sl::vpiIntegerTypespec => {
                let is_signed = typespec.vpi_get(sl::vpiSigned) == 1;

                self.ast.add_typ(Type {
                    token,
                    kind: TypeKind::Integer,
                    is_signed,
                })
            }
            sl::vpiLogicTypespec => {
                let is_signed = typespec.vpi_get(sl::vpiSigned) == 1;

                let ranges_array = typespec
                    .vpi_iter(sl::vpiRange)
                    .map(|r| self.translate_range(r))
                    .collect::<Vec<_>>();

                let range = if ranges_array.is_empty() {
                    None
                } else if ranges_array.len() == 1 {
                    Some(ranges_array[0].clone())
                } else {
                    self.err_unsupported(&token, format_args!("Multiple ranges are unsupported"));
                    None
                };

                self.ast.add_typ(Type {
                    token,
                    kind: TypeKind::Logic(range),
                    is_signed,
                })
            }

            sl::vpiBitTypespec => {
                let is_signed = typespec.vpi_get(sl::vpiSigned) == 1;

                let ranges_array = typespec
                    .vpi_iter(sl::vpiRange)
                    .map(|r| self.translate_range(r))
                    .collect::<Vec<_>>();

                let range = if ranges_array.is_empty() {
                    None
                } else if ranges_array.len() == 1 {
                    Some(ranges_array[0].clone())
                } else {
                    self.err_unsupported(&token, format_args!("Multiple ranges are unsupported"));
                    None
                };

                self.ast.add_typ(Type {
                    token,
                    kind: TypeKind::Bit(range),
                    is_signed,
                })
            }

            _ => {
                self.err_other(&token, format_args!("typespec {typ}"));
                self.ast.add_typ(Type {
                    token,
                    kind: TypeKind::Invalid,
                    is_signed: false,
                })
            }
        }
    }

    fn translate_constant(&mut self, constant: VpiHandle) -> ConstantIdx {
        let token = token_from_vpi(constant);
        let size = constant.vpi_get(sl::vpiSize);

        let kind = match constant.vpi_get_value() {
            VpiValue::BinaryString(val) => ConstantKind::Value {
                vals: str_to_values_le(&val, 2),
            },
            VpiValue::OctalString(val) => ConstantKind::Value {
                vals: str_to_values_le(&val, 8),
            },
            VpiValue::DecimalString(val) => ConstantKind::Value {
                vals: str_to_values_le(&val, 10),
            },
            VpiValue::HexadecimalString(val) => ConstantKind::Value {
                vals: str_to_values_le(&val, 16),
            },

            VpiValue::IntValue(val) => ConstantKind::Integer(val),
            VpiValue::UintValue(val) => ConstantKind::UnsignedInteger(val),

            VpiValue::RealValue(_) => {
                self.err_unsupported(&token, format_args!("constant real value"));
                ConstantKind::Invalid
            }
            VpiValue::StringValue(_) => {
                self.err_unsupported(&token, format_args!("constant string"));
                ConstantKind::Invalid
            }
            VpiValue::TimeValue(_) => {
                self.err_unsupported(&token, format_args!("constant time"));
                ConstantKind::Invalid
            }

            VpiValue::ObjectTypeValue => {
                self.err_unsupported(&token, format_args!("constant object"));
                ConstantKind::Invalid
            }
        };

        self.ast.add_constant(Constant { token, size, kind })
    }

    fn translate_expr_into_u32(&mut self, expr: VpiHandle) -> Option<u32> {
        let token = token_from_vpi(expr);

        let expr_idx = self.translate_expr(expr);
        let ast_expr = self.ast.get_expr(expr_idx);

        let ExprKind::Constant {
            constant: const_idx,
        } = ast_expr.kind
        else {
            self.err_unsupported(
                &token,
                format_args!("only constants are supported as range indices"),
            );
            return None;
        };

        match evaluate_constant_to_u32(self.ast.get_constant(const_idx)) {
            Ok(value) => Some(value),
            Err(err) => {
                self.errors.push(err);
                None
            }
        }
    }

    fn translate_range(&mut self, range: VpiHandle) -> Range {
        let lhs_vpi = range
            .vpi_handle(sl::vpiLeftRange)
            .expect("range has no left range");

        let rhs_vpi = range
            .vpi_handle(sl::vpiRightRange)
            .expect("range has no left range");

        // NOTE: It's the job of the translate function to put the error in the error
        //       array, we do it this way so we can report errors for both lhs and rhs.
        let left = self.translate_expr_into_u32(lhs_vpi).unwrap_or(0);
        let right = self.translate_expr_into_u32(rhs_vpi).unwrap_or(0);

        Range { left, right }
    }

    fn get_signal_from_ref(&self, ref_obj: VpiHandle) -> Option<SignalIdx> {
        let full_name = ref_obj.vpi_str(sl::vpiFullName);

        let scope = self.ast.get_scope(
            self.current_scope
                .expect("Getting a signal outside a scope"),
        );

        scope.find_signal_recursively(&self.ast, &full_name)
    }

    fn add_signal_to_current_scope(&mut self, signal_idx: SignalIdx) {
        let scope_idx = self.current_scope.expect("Adding a signal outside a scope");
        self.ast.get_scope_mut(scope_idx).signals.push(signal_idx);
    }

    fn push_scope(&mut self, scope: ScopeIdx) {
        self.current_scope = Some(scope);
    }

    fn pop_scope(&mut self) {
        let scope = self.ast.get_scope(
            self.current_scope
                .expect("Popping scope when no scope is defined"),
        );

        self.current_scope = Some(
            scope
                .parent
                .expect("Popping scope when scope has no parent scope"),
        );
    }

    fn add_to_block_end(
        &mut self,
        maybe_block: StatementIdx,
        statement_to_add: StatementIdx,
    ) -> StatementIdx {
        let statement = self.ast.get_statement_mut(maybe_block);

        match &mut statement.kind {
            StatementKind::Block {
                statements,
                scope: _,
            } => {
                statements.push(statement_to_add);
                return maybe_block;
            }
            _ => {}
        };

        let token = statement.token.clone();
        // Create new block
        let statements = vec![maybe_block, statement_to_add];

        self.ast.add_statement(Statement {
            token,
            kind: StatementKind::Block {
                statements,
                scope: self.current_scope.expect("No scope found"),
            },
        })
    }

    fn make_temp_signal(&mut self, typ: TypeIdx) -> SignalIdx {
        let scope = self.current_scope.expect("Creating signal outside scope");
        let name = format!("__tmp_frontend_{}", self.temp_counter);
        self.temp_counter += 1;

        let signal = self.ast.add_signal(Signal {
            token: Token {
                line: 0,
                name: name.clone(),
            },
            typ,
            lifetime: SignalLifetime::Automatic,
            full_name: name,
            scope,
        });

        self.ast.get_scope_mut(scope).signals.push(signal);

        signal
    }

    // TODO: We should probably make this fixed and not recreate it every time.
    fn make_int_typ(&mut self) -> TypeIdx {
        let token = Token {
            line: 0,
            name: "int".to_string(),
        };

        self.ast.add_typ(Type {
            token,
            kind: TypeKind::Int,
            is_signed: false,
        })
    }

    // TODO: We should probably make this fixed and not recreate it every time.
    fn make_constant_true(&mut self) -> ExprIdx {
        let token = Token {
            line: 0,
            name: "1".to_string(),
        };
        let constant = self.ast.add_constant(Constant {
            token: token.clone(),
            size: 32,
            kind: ConstantKind::Integer(1),
        });
        self.ast.add_expr(Expr {
            token,
            kind: ExprKind::Constant { constant },
        })
    }

    fn make_constant_zero(&mut self) -> ExprIdx {
        let token = Token {
            line: 0,
            name: "0".to_string(),
        };
        let constant = self.ast.add_constant(Constant {
            token: token.clone(),
            size: 32,
            kind: ConstantKind::Integer(0),
        });
        self.ast.add_expr(Expr {
            token,
            kind: ExprKind::Constant { constant },
        })
    }
    fn err_unsupported(&mut self, token: &Token, what: fmt::Arguments<'_>) {
        self.errors.push(FrontendError::unsupported(
            token.clone(),
            format!("Unsupported {what}"),
        ))
    }

    fn err_other(&mut self, token: &Token, what: fmt::Arguments<'_>) {
        self.errors
            .push(FrontendError::other(token.clone(), format!("{what}")))
    }

    fn err_signal_not_found(&mut self, ref_obj: VpiHandle) {
        let token = token_from_vpi(ref_obj);
        let signal_name = ref_obj.vpi_str(sl::vpiName);
        self.errors.push(FrontendError::other(
            token,
            format!("Signal \"{signal_name}\" not found."),
        ));
    }
}
