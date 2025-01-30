# Cudalator

Project for the course GPU Programming, a.y. 2024/2025.

## Table of contents

1. [How to build](#how-to-build)
2. [Docs](#docs)
4. [Libraries](#libraries-used)
3. [Dependencies](#dependencies)


## How to build

```sh
git submodule update --init --depth 1 --recursive
pip install -r requirements.txt

# Compile Surelog and setup the libraries
make

# Runs on a sample circuit
cargo run
```

## Docs

Docs are available at [https://daemondeal.github.io/cudalator/](https://daemondeal.github.io/cudalator/). You can also read them locally by running run `mkdocs serve`.


If you are a contributor, run `mkdocs gh-deploy` whenever you modify the docs.

## Libraries Used

- [Surelog](https://github.com/chipsalliance/Surelog)


## Dependencies

- CMake (version 3.21 or more)
- Rust and Cargo 
- CUDA
- Python 3
- Probably more

