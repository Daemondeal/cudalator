use std::collections::HashMap;

use crate::cir::{
    Ast, Expr, ExprIdx, ExprKind, ScopeIdx, Signal, SignalIdx, SignalLifetime, Statement,
    StatementIdx, StatementKind, Token, TypeIdx,
};

pub fn run_pass_remove_blocking_assignments(ast: &mut Ast) {
    let mut pass = PassRemoveBlockingAssignments::new();
    pass.run(ast);
}

struct PassRemoveBlockingAssignments {
    temp_counter: usize,
    temporaries: HashMap<SignalIdx, SignalIdx>,
    is_in_top_level_block: bool,
}

impl PassRemoveBlockingAssignments {
    pub fn new() -> Self {
        Self {
            temp_counter: 0,
            temporaries: HashMap::new(),
            is_in_top_level_block: false,
        }
    }

    pub fn run(&mut self, ast: &mut Ast) {
        let top_idx = ast.top_module.expect("No top module found");

        let top_module = ast.get_module(top_idx);
        let top_scope = top_module.scope;

        // FIXME: Rust's borrow checker at work. There is probably a better way to do this, but for
        //        now we'll just clone everything and hope it turns out okay.
        let processes = top_module.processes.clone();

        // The general strategy is to assign signals to temporary variables, and perform the actual
        // assignation only at the end of the scope
        for process_idx in processes {
            let process_stmt = ast.get_process(process_idx).statement;

            let mut is_block = false;
            let statement = ast.get_statement_mut(process_stmt);

            match &mut statement.kind {
                StatementKind::SimpleAssignment { blocking, .. }
                | StatementKind::SimpleAssignmentParts { blocking, .. } => {
                    // If the process only has a single blocking assignment, then it's fine to make it non-blocking.
                    *blocking = false;
                }
                StatementKind::Block {
                    statements: _,
                    scope: _,
                } => {
                    is_block = true;
                }
                _ => {}
            }

            if is_block {
                let statements = match &ast.get_statement(process_stmt).kind {
                    StatementKind::Block {
                        statements,
                        scope: _,
                    } => statements.clone(),
                    _ => unreachable!(),
                };

                self.is_in_top_level_block = true;
                for statement in statements {
                    self.process_statement(ast, statement, top_scope);
                }

                let mut assignments = vec![];

                for (original, temp) in &mut self.temporaries {
                    let temp_ref = Self::make_signal_ref(ast, *temp);
                    assignments.push(Self::make_assignment(ast, *original, temp_ref));

                    // let orig_ref = Self::make_signal_ref(ast, *original);
                    // assignments.push(ast.add_statement(Statement {
                    //     token: Token::dummy(),
                    //     kind: StatementKind::Assignment {
                    //         lhs: orig_ref,
                    //         rhs: temp_ref,
                    //         blocking: false,
                    //     },
                    // ));
                }
                self.temporaries.clear();

                match &mut ast.get_statement_mut(process_stmt).kind {
                    StatementKind::Block {
                        statements,
                        scope: _,
                    } => {
                        statements.append(&mut assignments);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn process_statement(&mut self, ast: &mut Ast, statement: StatementIdx, scope: ScopeIdx) {
        let (should_patch, typ) = match &ast.get_statement(statement).kind {
            StatementKind::SimpleAssignment {
                target,
                source: _,
                blocking,
            }
            | StatementKind::SimpleAssignmentParts {
                target,
                source: _,
                from: _,
                to: _,
                blocking,
            } => (
                ast.get_signal(*target).lifetime == SignalLifetime::Static && *blocking,
                Some(ast.get_signal(*target).typ),
            ),

            _ => (false, None),
        };

        let statement = ast.get_statement_mut(statement);

        match &mut statement.kind {
            StatementKind::Invalid => unreachable!(),
            StatementKind::Assignment {
                lhs: _,
                rhs: _,
                blocking: _,
            } => unreachable!(),

            StatementKind::Block { statements, scope } => {
                let scope = *scope;

                for statement in statements.clone() {
                    self.process_statement(ast, statement, scope);
                }

                let signals = ast.get_scope(scope).signals.clone();

                let mut additional_statements = vec![];
                for signal in signals {
                    if let Some(replacement) = self.temporaries.remove(&signal) {
                        // let replacement = *self.temporaries.get(&signal).unwrap();
                        let replacement_ref = Self::make_signal_ref(ast, replacement);

                        additional_statements.push(Self::make_assignment(
                            ast,
                            signal,
                            replacement_ref,
                        ));
                    }
                }
            }

            StatementKind::SimpleAssignment {
                target,
                source,
                blocking,
            } => {
                let source = *source;
                

                let to_add = if should_patch {
                    *blocking = false;
                    if let Some(renamed) = self.temporaries.get(target) {
                        *target = *renamed;
                        None
                    } else {
                        Some((*target, self.add_temp_signal(ast, typ.unwrap(), scope)))
                    }
                } else {
                    None
                };

                self.patch_expr(ast, source);

                if let Some((original, modified)) = to_add {
                    self.temporaries.insert(original, modified);
                }
            }

            StatementKind::SimpleAssignmentParts {
                target,
                source,
                from,
                to,
                blocking: _,
            } => {
                let source = *source;
                let from = *from;
                let to = *to;

                let to_add = if should_patch {
                    if let Some(renamed) = self.temporaries.get(target) {
                        *target = *renamed;
                        None
                    } else {
                        Some((*target, self.add_temp_signal(ast, typ.unwrap(), scope)))
                    }
                } else {
                    None
                };

                self.patch_expr(ast, source);
                self.patch_expr(ast, from);
                self.patch_expr(ast, to);

                if let Some((original, modified)) = to_add {
                    self.temporaries.insert(original, modified);
                }
            }

            StatementKind::While { condition, body }
            | StatementKind::DoWhile { condition, body } => {
                let condition = *condition;
                let body = *body;

                self.patch_expr(ast, condition);
                self.process_statement(ast, body, scope);
            }

            StatementKind::If {
                condition,
                body,
                else_,
            } => {
                let condition = *condition;
                let body = *body;
                let else_ = *else_;

                self.patch_expr(ast, condition);
                self.process_statement(ast, body, scope);
                if let Some(else_) = else_ {
                    self.process_statement(ast, else_, scope);
                }
            }

            StatementKind::Case => todo!("PassRemoveBlockingAssignments case"),
            StatementKind::Foreach => todo!("PassRemoveBlockingAssignments foreach"),
            StatementKind::Return { expr } => {
                let expr = *expr;
                self.patch_expr(ast, expr);
            }
            StatementKind::Break => {}
            StatementKind::Continue => {}
            StatementKind::Null => {}
        }
    }

    fn patch_expr(&mut self, ast: &mut Ast, expr: ExprIdx) {
        // First Pass: modify children
        match &mut ast.get_expr_mut(expr).kind {
            ExprKind::Invalid => unreachable!(),
            ExprKind::Constant { constant: _ } => {}
            ExprKind::Unary { op: _, expr } => {
                let expr = *expr;
                self.patch_expr(ast, expr);
            }

            ExprKind::Binary { op: _, lhs, rhs } => {
                // Copy values early
                let (lhs, rhs) = (*lhs, *rhs);

                self.patch_expr(ast, lhs);
                self.patch_expr(ast, rhs);
            }

            ExprKind::Concatenation { exprs } => {
                for expr in exprs.clone() {
                    self.patch_expr(ast, expr);
                }
            }

            ExprKind::PartSelect { lhs, rhs, target } => {
                let (lhs, rhs) = (*lhs, *rhs);

                if let Some(replacement) = self.temporaries.get(target) {
                    *target = *replacement;
                }

                self.patch_expr(ast, lhs);
                self.patch_expr(ast, rhs);
            }
            ExprKind::BitSelect { expr, target } => {
                if let Some(replacement) = self.temporaries.get(target) {
                    *target = *replacement;
                }

                let expr = *expr;

                self.patch_expr(ast, expr);
            }
            ExprKind::SignalRef { signal } => {
                if let Some(replacement) = self.temporaries.get(signal) {
                    *signal = *replacement;
                }
            }
        }
    }

    fn make_assignment(ast: &mut Ast, target: SignalIdx, source: ExprIdx) -> StatementIdx {
        ast.add_statement(Statement {
            token: Token::dummy(),
            kind: StatementKind::SimpleAssignment {
                target,
                source,
                blocking: false,
            },
        })
    }

    fn make_signal_ref(ast: &mut Ast, signal: SignalIdx) -> ExprIdx {
        ast.add_expr(Expr {
            token: Token::dummy(),
            kind: ExprKind::SignalRef { signal },
        })
    }

    fn add_temp_signal(&mut self, ast: &mut Ast, typ: TypeIdx, scope: ScopeIdx) -> SignalIdx {
        let name = format!("__tmp_pass_pba_{}", self.temp_counter);
        self.temp_counter += 1;

        let token = Token {
            line: 0,
            name: name.clone(),
        };

        ast.add_signal(Signal {
            token,
            typ,
            lifetime: SignalLifetime::Automatic,
            full_name: name,
            scope,
        })
    }
}
