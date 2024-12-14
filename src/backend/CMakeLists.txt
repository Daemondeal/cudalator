add_library(
    cudalator-backend
    ./CirPrinter.cpp
    ./CirPrinter.hpp
)

target_include_directories(
    cudalator-backend
    PRIVATE ${CMAKE_SOURCE_DIR}/build/libraries/include
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

