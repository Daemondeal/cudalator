cmake_minimum_required(VERSION 3.20)
project(
    cudalator
    VERSION 0.1.0
    LANGUAGES CXX)

link_directories(${CMAKE_SOURCE_DIR}/libraries/lib)

add_executable(cudalator src/main.cpp)

target_include_directories(
    cudalator
    PRIVATE ${CMAKE_SOURCE_DIR}/libraries/include
)

target_link_libraries(
    cudalator
    PRIVATE surelog uhdm antlr4-runtime capnp kj z
)
