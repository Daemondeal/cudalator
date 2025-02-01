pub mod cir;

mod cir_printer;

mod backend;
mod frontend;

use clap::Parser;
use log::error;

use crate::backend::pass_manager;
use crate::frontend::sv_frontend;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(index = 1, required = true, help = "The input SystemVerilog files")]
    files: Vec<String>,

    #[arg(short, long, help = "Prints the AST after translation and running all passes")]
    print_ast: bool,

    // TODO: Check if the  default is correct
    #[arg(long = "top", help = "Specify the top module to use. Defaults to the first one it sees")]
    top_module: Option<String>,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    env_logger::builder().format_timestamp(None).init();

    let args = Args::parse();

    run_compiler_pipeline(args);

    Ok(())
}

fn run_compiler_pipeline(args: Args) {
    let maybe_ast = sv_frontend::compile_systemverilog(&args.files, args.top_module.as_deref());

    let mut ast = match maybe_ast {
        Ok(ast) => ast,
        Err(errors) => {
            for err in errors {
                error!("line {}: {}", err.token.line, err.message);
            }
            return;
        }
    };

    pass_manager::run_passes(&mut ast);

    if args.print_ast {
        cir_printer::print_cir_ast(&ast, &mut std::io::stdout()).unwrap();
    }
}
