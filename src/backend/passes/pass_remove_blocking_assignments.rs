use std::collections::HashMap;

use crate::cir::{
    Ast, ScopeIdx, SignalIdx, StatementIdx, StatementKind,
};

pub fn run_pass_remove_blocking_assignments(ast: &mut Ast) {
    let mut pass = PassRemoveBlockingAssignments::new();
    pass.run(ast);
}

struct PassRemoveBlockingAssignments {
    temp_uuid: usize,
    temporaries: HashMap<SignalIdx, SignalIdx>,
    is_in_top_level_block: bool,
}

impl PassRemoveBlockingAssignments {
    pub fn new() -> Self {
        Self {
            temp_uuid: 0,
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
            let process = ast.get_process(process_idx);

            self.is_in_top_level_block = true;
            self.process_statement(ast, process.statement);
        }
    }

    pub fn process_statement(
        &mut self,
        ast: &mut Ast,
        statement: StatementIdx,
    ) {
        let statement = ast.get_statement_mut(statement);

        match &mut statement.kind {
            StatementKind::Invalid => unreachable!(),
            StatementKind::Assignment { lhs, rhs, blocking } => todo!(),
            StatementKind::Block { statements } => todo!(),
            StatementKind::ScopedBlock { statements, scope } => todo!(),
            StatementKind::If { condition, body } => todo!(),
            StatementKind::IfElse { condition, body, else_ } => todo!(),
            StatementKind::While { condition, body } => todo!(),
            StatementKind::DoWhile { condition, body } => todo!(),
            StatementKind::Repeat { condition, body } => todo!(),
            StatementKind::For { condition, init, increment, body, scope } => todo!(),
            StatementKind::Case => todo!(),
            StatementKind::Foreach => todo!(),
            StatementKind::Forever { body } => todo!(),
            StatementKind::Return { expr } => todo!(),
            StatementKind::Break => todo!(),
            StatementKind::Continue => todo!(),
            StatementKind::Null => todo!(),
        }
    }
}
