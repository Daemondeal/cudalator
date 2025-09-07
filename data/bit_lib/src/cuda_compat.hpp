#pragma once

#ifdef __CUDACC__
#include <cstdio> // for printf/fprintf
#include <cuda_runtime.h>

#define HOST_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#else // when compiling for a cpu..
#define HOST_DEVICE
#define HOST
#define DEVICE
#define CUDA_CHECK(err) err

#endif
