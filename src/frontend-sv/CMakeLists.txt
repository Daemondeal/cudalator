add_library(
    cudalator-frontend-sv
    ./SampleListener.hpp
    ./SampleListener.cpp
    ./Entry.cpp
    ./Entry.hpp
)

link_directories(${CMAKE_SOURCE_DIR}/build/libraries/lib)

target_include_directories(
    cudalator-frontend-sv
    PRIVATE ${CMAKE_SOURCE_DIR}/build/libraries/include
)

target_link_libraries(
    cudalator-frontend-sv
    PRIVATE surelog uhdm antlr4-runtime capnp kj z
)
