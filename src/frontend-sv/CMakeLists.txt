add_library(
    cudalator-frontend-sv
    ./utils.cpp
    ./utils.hpp
    ./SampleListener.hpp
    ./SampleListener.cpp
    ./SurelogParser.cpp
    ./SurelogParser.hpp
    ./SystemVerilogFrontend.cpp
    ./SystemVerilogFrontend.hpp
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
