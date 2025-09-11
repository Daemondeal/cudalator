cmake_minimum_required(VERSION 3.21)

# default to OFF
option(ENABLE_CUDA "Enable CUDA for GPU acceleration" OFF)

if(ENABLE_CUDA)
    project(run_simulation VERSION 0.1.0 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 53)
else()
    project(run_simulation VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Conditionally set source file variables for BOTH the module and the runtime
if(ENABLE_CUDA)
    set(GENERATED_SOURCE_FILE src/codegen/module.cu)
    set(RUNTIME_SOURCE_FILE   src/runtime/Runtime.cu)
    set(MAIN_SOURCE_FILE      src/main.cu)
else()
    set(GENERATED_SOURCE_FILE src/codegen/module.cpp)
    set(RUNTIME_SOURCE_FILE   src/runtime/Runtime.cpp)
    set(MAIN_SOURCE_FILE      src/main.cpp) # <-- And this
endif()

# Use the variables to add the correct source files to the executable
add_executable(
    run_simulation
    ${MAIN_SOURCE_FILE}
    ${GENERATED_SOURCE_FILE}
    ${RUNTIME_SOURCE_FILE}
    src/runtime/Stats.cpp
    src/libs/format.cc
    src/libs/os.cc
)

target_include_directories(run_simulation PUBLIC src/libs)
target_include_directories(run_simulation PUBLIC src/libs/fmt)
target_link_libraries(run_simulation)

# Disable some warnings
if(ENABLE_CUDA)
    list(APPEND CUDA_NVCC_FLAGS --diag-suppress 128)
    list(APPEND CUDA_NVCC_FLAGS --diag-suppress 2417)

    # Attach to target
    # NOTE: `SHELL:` is needed otherwise CMake performs auto option deduplication,
    #       putting "-option A" "-option B" as "-option A B"
    target_compile_options(run_simulation PRIVATE
        "SHELL:$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>"
    )
endif()
