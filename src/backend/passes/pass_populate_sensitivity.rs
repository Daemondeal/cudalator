use std::collections::HashSet;

use crate::cir::{Ast, ExprIdx, ExprKind, SensitivtyKind, SignalIdx, StatementIdx, StatementKind};

pub fn run_pass_populate_sensitivity(ast: &mut Ast) {
    let top_idx = ast.top_module.expect("No top module found");

    let top_module = ast.get_module(top_idx);

    // FIXME: Rust's borrow checker at work. There is probably a better way to do this, but for
    //        now we'll just clone everything and hope it turns out okay.
    let top_scope_signals = ast.get_scope(top_module.scope).signals.clone();
    let processes = top_module.processes.clone();

    for process_idx in processes {
        let mut signal_set = HashSet::new();

        let process = ast.get_process(process_idx);
        if !process.should_populate_sensitivity_list {
            continue;
        }
        for statement in process.statements.clone() {
            collect_signals_statement(&ast, &mut signal_set, &top_scope_signals, statement);
        }

        let process = ast.get_process_mut(process_idx);

        for (kind, idx) in process.sensitivity_list.iter_mut() {
            if signal_set.contains(idx) {
                *kind = SensitivtyKind::OnChange;
                signal_set.remove(idx);
            }
        }

        for signal in signal_set {
            process
                .sensitivity_list
                .push((SensitivtyKind::OnChange, signal));
        }
    }
}

fn collect_signals_statement(
    ast: &Ast,
    signals: &mut HashSet<SignalIdx>,
    top_scope_signals: &[SignalIdx],
    statement_idx: StatementIdx,
) {
    let statement = ast.get_statement(statement_idx);

    match &statement.kind {
        StatementKind::Assignment { lhs: _, rhs, .. } => {
            collect_signals_expr(ast, signals, top_scope_signals, *rhs);
        }
        StatementKind::Block {
            statements,
            scope: _,
        } => {
            for statement in statements {
                collect_signals_statement(ast, signals, top_scope_signals, *statement);
            }
        }
        StatementKind::If {
            condition,
            body,
            else_,
        } => {
            collect_signals_expr(ast, signals, top_scope_signals, *condition);
            collect_signals_statement(ast, signals, top_scope_signals, *body);
            if let Some(else_) = else_ {
                collect_signals_statement(ast, signals, top_scope_signals, *else_);
            }
        }

        StatementKind::While { condition, body } => {
            collect_signals_expr(ast, signals, top_scope_signals, *condition);
            collect_signals_statement(ast, signals, top_scope_signals, *body);
        }

        StatementKind::DoWhile { condition, body } => {
            collect_signals_expr(ast, signals, top_scope_signals, *condition);
            collect_signals_statement(ast, signals, top_scope_signals, *body);
        }

        StatementKind::Return { expr } => {
            collect_signals_expr(ast, signals, top_scope_signals, *expr)
        }
        StatementKind::Break => {}
        StatementKind::Continue => {}
        StatementKind::Null => {}
        StatementKind::Invalid => {}

        StatementKind::Case => todo!(),
        StatementKind::Foreach => todo!(),
    };
}

fn collect_signals_expr(
    ast: &Ast,
    signals: &mut HashSet<SignalIdx>,
    top_scope_signals: &[SignalIdx],
    expr_idx: ExprIdx,
) {
    match &ast.get_expr(expr_idx).kind {
        ExprKind::Invalid => {}
        ExprKind::Constant { constant: _ } => {}
        ExprKind::Unary { op: _, expr } => {
            collect_signals_expr(ast, signals, top_scope_signals, *expr);
        }
        ExprKind::Binary { op: _, lhs, rhs } => {
            collect_signals_expr(ast, signals, top_scope_signals, *lhs);
            collect_signals_expr(ast, signals, top_scope_signals, *rhs);
        }
        ExprKind::Concatenation { exprs } => {
            for expr in exprs {
                collect_signals_expr(ast, signals, top_scope_signals, *expr);
            }
        }
        ExprKind::PartSelect { lhs, rhs, target } => {
            collect_signals_expr(ast, signals, top_scope_signals, *lhs);
            collect_signals_expr(ast, signals, top_scope_signals, *rhs);

            if signal_is_in_list(top_scope_signals, *target) {
                signals.insert(*target);
            }
        }
        ExprKind::BitSelect { expr, target } => {
            collect_signals_expr(ast, signals, top_scope_signals, *expr);
            if signal_is_in_list(top_scope_signals, *target) {
                signals.insert(*target);
            }
        }
        ExprKind::SignalRef { signal } => {
            if signal_is_in_list(top_scope_signals, *signal) {
                signals.insert(*signal);
            }
        }
    };
}

fn signal_is_in_list(signals: &[SignalIdx], signal_idx: SignalIdx) -> bool {
    for signal in signals {
        if signal.get_idx() == signal_idx.get_idx() {
            return true;
        }
    }

    false
}
