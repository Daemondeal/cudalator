use crate::cir::Ast;

use super::passes::pass_populate_sensitivity::run_pass_populate_sensitivity;

pub fn run_passes(ast: &mut Ast) {
    run_pass_populate_sensitivity(ast);
}
