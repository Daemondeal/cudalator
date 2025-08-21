use std::env;
use std::path::PathBuf;

fn main() {
    let src_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Build the C++ wrapper using the cc crate. This should use whatever C compiler is
    // available in the host machine, and automatically link the result with the final rust
    // binary.
    cc::Build::new()
        .cpp(true)
        .file("./surelog_cpp_wrapper/surelog_cpp_wrapper.cpp")
        // This flag is needed because otherwise the compiler complains about some unused
        // parameters in the source code of surelog itself, which is not ideal.
        .flag("-std=c++17")
        .flag("-Wno-unused-parameter")
        .flag("-std=c++17")
        .flag("-I./build/surelog/include/")
        .flag("-L./build/surelog/lib/")
        .flag("-lsurelog")
        .flag("-luhdm")
        .flag("-lantlr4-runtime")
        .flag("-lcapnp")
        .flag("-lkj")
        .flag("-lz")
        .compile("surelog_wrapper");

    // Make sure to rebuild in case the wrapper changes.
    println!("cargo:rerun-if-changed=surelog_cpp_wrapper/surelog_cpp_wrapper.cpp");

    // Link with the libraries that Surelog itself needs to work.
    println!(
        "cargo:rustc-link-search=native={}/build/surelog/lib",
        src_dir
    );
    println!("cargo:rustc-link-lib=static=surelog");
    println!("cargo:rustc-link-lib=static=uhdm");
    println!("cargo:rustc-link-lib=static=antlr4-runtime");
    println!("cargo:rustc-link-lib=static=capnp");
    println!("cargo:rustc-link-lib=static=kj");
    println!("cargo:rustc-link-lib=z");

    // Automatically generate the bindings for the C++ library using bindgen.
    let bindings = bindgen::Builder::default()
        .clang_arg("-I./build/surelog/include")
        .header("./surelog_cpp_wrapper/surelog_cpp_wrapper.hpp")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
