# Cudalator

Project for the course GPU Programming, a.y. 2024/2025.

## Table of contents

1. [How to build](#how-to-build)
2. [Docs](#docs)
3. [Tools](#tools)
4. [Libraries Used](#libraries-used)
5. [Dependencies](#dependencies)


## How to build

```sh
git submodule update --init --depth 1 --recursive
pip install -r requirements.txt

# Compile Surelog and setup the libraries
make

# Runs on a sample circuit
./tools/run_demo.sh cpu ./demos/multiplier/

# Run the same demo but on GPU
./tools/run_demo.sh gpu ./demos/multiplier/
```

## Docs

Docs are available at [https://daemondeal.github.io/cudalator/](https://daemondeal.github.io/cudalator/). You can also read them locally by running run `mkdocs serve`.


If you are a contributor, run `mkdocs gh-deploy` whenever you modify the docs.

## Tools

The project includes some tools useful for contributors. They are all found in the tools subdirectory, and they expect to be called from the root folder.

These tools are:
- `./tools/run_tester.sh`: Runs a systemverilog file with Verilator, Icarus Verilog and Questasim, then shows their outputs. Useful to check the expected behavior of a SV construct.
- `./tools/run_demo.sh`: Given a folder, it will compile every `.sv` file inside with Cudalator, then copy either `main.cpp` or `main.cu` inside the resulting folder, then run the demo. Try the demos inside `./demos`

## Libraries Used

- [Surelog](https://github.com/chipsalliance/Surelog)
- [fmt](https://github.com/fmtlib/fmt)


## Dependencies

- CMake (version 3.21 or more)
- Rust and Cargo 
- CUDA
- Python 3
- Probably more

