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

make
```

You can run `make run` to run the code on an example circuit

## Docs

Docs are available at [https://daemondeal.github.io/cudalator/](https://daemondeal.github.io/cudalator/). You can also read them locally by running run `mkdocs serve`.


If you are a contributor, run `mkdocs gh-deploy` whenever you modify the docs.

## Libraries Used

- [Surelog](https://github.com/chipsalliance/Surelog)
- [spdlog](https://github.com/gabime/spdlog)
- [fmt 11.0.2](https://github.com/fmtlib/fmt)
- [argparse 3.1](https://github.com/p-ranav/argparse)


## Dependencies

- CMake (version 3.21 or more)
- Ninja
- CUDA
- Python 3
- Probably more

