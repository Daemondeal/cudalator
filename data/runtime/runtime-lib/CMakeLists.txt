add_library(cudalator-runtime-lib
    VLSignal.cpp
    VLSignal.hpp
)
target_include_directories(cudalator-runtime-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
