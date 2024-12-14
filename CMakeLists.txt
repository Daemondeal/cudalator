cmake_minimum_required(VERSION 3.20)
project(
    cudalator-compiler
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_directories(${CMAKE_SOURCE_DIR}/build/libraries/lib)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    include_directories(${CMAKE_SOURCE_DIR}/external/spdlog/include)
endif()

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
    PRIVATE spdlog::spdlog
)
