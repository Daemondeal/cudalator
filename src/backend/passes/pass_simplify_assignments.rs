use crate::cir::{Ast, ExprKind, StatementIdx, StatementKind};

pub fn run_pass_simplify_assignments(ast: &mut Ast) {
    let top_idx = ast.top_module.expect("No top module found");

    let top_module = ast.get_module(top_idx);

    for process_idx in top_module.processes.clone() {
        let statement = ast.get_process(process_idx).statement;
        simplify_statement(ast, statement);
    }
}

fn simplify_statement(ast: &mut Ast, statement: StatementIdx) {
    let kind = &mut ast.get_statement_mut(statement).kind;

    // Pass 1: replace children
    match kind {
        StatementKind::Assignment { lhs: _, rhs: _, blocking: _ } => {}
        StatementKind::Block {
            statements,
            scope: _,
        } => {
            for statement in statements.clone() {
                simplify_statement(ast, statement);
            }
        }


        StatementKind::If {
            condition: _,
            body,
            else_,
        } => {
            let body = *body;
            let else_ = *else_;

            simplify_statement(ast, body);
            if let Some(else_) = else_ {
                simplify_statement(ast, else_);
            }
        }

        StatementKind::While { condition: _, body }
        | StatementKind::DoWhile { condition: _, body } => {
            let body = *body;
            simplify_statement(ast, body);
        }

        StatementKind::Return { expr: _ } => {}
        StatementKind::Break => {}
        StatementKind::Continue => {}
        StatementKind::Null => {}

        StatementKind::Case => todo!(),
        StatementKind::Foreach => todo!(),

        StatementKind::Invalid => unreachable!(),

        StatementKind::SimpleAssignmentParts {
            target: _,
            source: _,
            from: _,
            to: _,
            blocking: _,
        } => unreachable!(),
        StatementKind::SimpleAssignment {
            target: _,
            source: _,
            blocking: _,
        } => unreachable!(),
    }

    // Pass 2: replace
    let replace_to = match ast.get_statement(statement).kind {
        StatementKind::Assignment { lhs, rhs, blocking } => {
            match ast.get_expr(lhs).kind {
                ExprKind::SignalRef { signal } => {
                    StatementKind::SimpleAssignment {
                        target: signal,
                        source: rhs,
                        blocking,
                    }
                }

                ExprKind::BitSelect { target, expr } => {
                    // NOTE: This will evaluate expr two times. Maybe fix this sometime.
                    StatementKind::SimpleAssignmentParts {
                        target,
                        from: expr,
                        to: expr,
                        source: rhs,
                        blocking,
                    }
                }
                ExprKind::PartSelect { target, lhs: lhs_, rhs: rhs_ } => {
                    // NOTE: This will evaluate expr two times. Maybe fix this sometime.
                    StatementKind::SimpleAssignmentParts {
                        target,
                        from: lhs_,
                        to: rhs_,
                        source: rhs,
                        blocking,
                    }
                }

                ExprKind::Concatenation { exprs: _ } => todo!("simplify_statement Concatenation"),

                _ => todo!("Invalid Assignment. TODO: Make this a proper error"),
            }
        }
        _ => return
    };


    ast.get_statement_mut(statement).kind = replace_to;
}
