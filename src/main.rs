pub mod cir;

mod cir_printer;

mod backend;
mod frontend;

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use backend::codegen::CodegenTarget;
use cir::Ast;
use clap::Parser;
use color_eyre::Result;
use log::error;

use crate::backend::pass_manager;
use crate::frontend::sv_frontend;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(index = 1, required = true, help = "The input SystemVerilog files")]
    files: Vec<String>,

    #[arg(
        short,
        long,
        help = "Prints the AST after translation and running all passes"
    )]
    print_ast: bool,

    // TODO: Check if the  default is correct
    #[arg(
        long = "top",
        help = "Specify the top module to use. Defaults to the first one it sees"
    )]
    top_module: Option<String>,

    #[arg(long = "output", short = 'o', help = "The output folder to codegen to")]
    output_folder: PathBuf,

    #[arg(long = "cpu", help = "Generate code meant to run on CPUs")]
    target_is_cpu: bool,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    env_logger::builder().format_timestamp(None).init();

    let args = Args::parse();

    let Some(ast) = run_compiler_pipeline(&args) else {
        std::process::exit(-1)
    };

    if args.print_ast {
        cir_printer::print_cir_ast(&ast, &mut std::io::stdout()).unwrap();
    }

    prepare_output_folder(&args)?;

    let path_header = args.output_folder.join("module.hpp");
    let path_source = if args.target_is_cpu {
        args.output_folder.join("module.cpp")
    } else {
        args.output_folder.join("module.cu")
    };

    codegen_into_files(&args, &ast, &path_header, &path_source)?;

    // FIXME: REMOVE THIS, THIS IS ONLY FOR DEBUG PURPOSES
    println!("{path_header:?}:");
    let header_contents = fs::read_to_string(&path_header)?;
    println!("{header_contents}");

    println!("{path_source:?}:");
    let source_contents = fs::read_to_string(&path_source)?;
    println!("{source_contents}");

    Ok(())
}

fn codegen_into_files(
    args: &Args,
    ast: &Ast,
    path_header: &Path,
    path_source: &Path,
) -> Result<()> {
    let file_header = File::create(path_header)?;
    let file_source = File::create(path_source)?;

    let mut writer_header = BufWriter::new(file_header);
    let mut writer_source = BufWriter::new(file_source);

    let target = if args.target_is_cpu {
        CodegenTarget::CPU
    } else {
        CodegenTarget::CUDA
    };

    let res =
        backend::codegen::codegen_into_files(ast, &mut writer_source, &mut writer_header, target);

    writer_header.flush()?;
    writer_source.flush()?;

    res
}

fn run_compiler_pipeline(args: &Args) -> Option<Ast> {
    let maybe_ast = sv_frontend::compile_systemverilog(&args.files, args.top_module.as_deref());

    let mut ast = match maybe_ast {
        Ok(ast) => ast,
        Err(errors) => {
            for err in errors {
                error!("line {}: {}", err.token.line, err.message);
            }
            return None;
        }
    };

    pass_manager::run_passes(&mut ast);

    Some(ast)
}

fn prepare_output_folder(args: &Args) -> Result<()> {
    if fs::exists(&args.output_folder)? {
        // TODO: Check that the required files are inside the folder already
    }

    fs::create_dir_all(&args.output_folder)?;

    Ok(())
}
