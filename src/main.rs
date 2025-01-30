pub mod cir;
mod frontend;
mod cir_printer;

use log::error;

use crate::frontend::sv_frontend;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    env_logger::builder()
        .format_timestamp(None)
        .init();

    let ast = sv_frontend::compile_systemverilog(&vec!["./data/rtl/adder.sv".to_owned()]);

    // TODO: Do this properly
    if let Err(errors) = ast {
        for err in errors {
            error!("line {}: {}", err.token.line, err.message);
        }

        return Ok(());
    }

    // TODO: Make this less ugly
    let ast = ast.ok().unwrap();

    let top = ast.get_module(ast.top_module.expect("No top module found"));
    println!("top: {}", top.token.name);

    cir_printer::print_cir_ast(&ast, &mut std::io::stdout()).unwrap();

    Ok(())
}
