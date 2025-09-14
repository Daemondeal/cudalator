cmake_minimum_required(VERSION 3.21)

# TODO: Add CUDA
project(
    run_simulation
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# TODO: Move to CUDA
add_executable(
    run_simulation
    src/main.cpp
    src/codegen/module.cpp
    src/runtime/Runtime.cpp
    src/runtime/Stats.cpp
    src/libs/format.cc
    src/libs/os.cc
)
target_include_directories(run_simulation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/libs")
target_link_libraries(run_simulation)
