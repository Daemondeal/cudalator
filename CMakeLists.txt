cmake_minimum_required(VERSION 3.20)
project(
    cudalator-compiler
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Winit-self -Wno-unused-parameter")

link_directories(${CMAKE_SOURCE_DIR}/build/libraries/lib)

include_directories(${CMAKE_SOURCE_DIR}/external/fmt/include/)

set(SPDLOG_FMT_EXTERNAL ON)

add_subdirectory(external/fmt/)

set(spdlog_DIR ${CMAKE_SOURCE_DIR}/external/spdlog/build/)
find_package(spdlog REQUIRED)

add_executable(
    cudalator-compiler
    src/main.cpp
)

target_include_directories(
    cudalator-compiler
    PRIVATE ${CMAKE_SOURCE_DIR}/build/libraries/include
)

target_include_directories(
    cudalator-compiler
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(src/frontend-sv)
add_subdirectory(src/backend)

target_link_libraries(
    cudalator-compiler
    PRIVATE cudalator-frontend-sv
    PRIVATE cudalator-backend
)

target_link_libraries(
    cudalator-compiler
    PRIVATE fmt::fmt
    PRIVATE spdlog::spdlog
)
