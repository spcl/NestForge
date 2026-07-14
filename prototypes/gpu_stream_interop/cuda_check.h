// Shared CUDA error-check macro for the gpu_stream_interop drivers (previously duplicated, and diverged,
// across driver.cpp / driver_control.cpp / omp_probe.cpp). CK(expr) runs a CUDA runtime call and, on
// error, prints the failing expression + message and returns 2 from the enclosing function. A macro is
// required here: it must early-return from the caller and stringize the expression, which a function cannot.
#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define CK(expr)                                                                                                                                               \
    do {                                                                                                                                                       \
        const cudaError_t err = (expr);                                                                                                                        \
        if (err != cudaSuccess) {                                                                                                                              \
            fprintf(stderr, "CUDA %s: %s\n", #expr, cudaGetErrorString(err));                                                                                  \
            return 2;                                                                                                                                          \
        }                                                                                                                                                      \
    } while (0)
