add_library(
    cudalator-backend
    ./Backend.cpp
    ./Backend.hpp
    ./CirPrinter.cpp
    ./CirPrinter.hpp
    ./PassManager.cpp
    ./PassManager.hpp
    ./passes/PassBase.hpp
    ./passes/PopulateSensitivityList.cpp
    ./passes/PopulateSensitivityList.hpp
)

target_include_directories(
    cudalator-backend
    PRIVATE ${CMAKE_SOURCE_DIR}/build/libraries/include
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

