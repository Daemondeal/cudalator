use crate::cir::Ast;

use super::passes::{pass_populate_sensitivity::run_pass_populate_sensitivity, pass_remove_blocking_assignments::run_pass_remove_blocking_assignments};

pub fn run_passes(ast: &mut Ast) {
    run_pass_populate_sensitivity(ast);
    run_pass_remove_blocking_assignments(ast);
}
