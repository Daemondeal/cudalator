cmake_minimum_required(VERSION 3.21)

# Using the {fmt} library to make demos easier to write
# NOTE: It's a bit ugly to add a dependency like this to a runtime,
#        but it's probably preferrable than requiring a C++23 compiler.
# FIXME: Maybe we should just remove this and deal with normal C++ iostream
include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

# TODO: Add CUDA

project(
    run_simulation
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO: Move to CUDA
add_executable(
    run_simulation
    src/main.cpp
    src/codegen/module.cpp
    src/runtime/Runtime.cpp
)

target_link_libraries(run_simulation fmt::fmt)
