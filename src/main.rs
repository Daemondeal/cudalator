pub mod cir;

mod cir_printer;

mod frontend;
mod backend;

use clap::Parser;
use log::error;

use crate::frontend::sv_frontend;
use crate::backend::pass_manager;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(index = 1, required = true)]
    files: Vec<String>,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    env_logger::builder()
        .format_timestamp(None)
        .init();

    let args = Args::parse();

    let ast = sv_frontend::compile_systemverilog(&args.files);

    // TODO: Do this properly
    if let Err(errors) = ast {
        for err in errors {
            error!("line {}: {}", err.token.line, err.message);
        }

        return Ok(());
    }

    // TODO: Make this less ugly
    let mut ast = ast.ok().unwrap();

    pass_manager::run_passes(&mut ast);


    let top = ast.get_module(ast.top_module.expect("No top module found"));
    println!("top: {}", top.token.name);

    cir_printer::print_cir_ast(&ast, &mut std::io::stdout()).unwrap();

    Ok(())
}
