cmake_minimum_required(VERSION 3.21)

# TODO: Add CUDA

add_executable(
    cudalator-runtime
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(runtime-lib)
add_subdirectory(codegen-example)

link_directories(${CMAKE_SOURCE_DIR}/libraries/lib)

add_executable(
    cudalator-runtime
    src/main.cpp
)

target_link_libraries(cudalator-runtime cudalator-runtime-lib)
target_link_libraries(cudalator-runtime cudalator-codegen-example)
