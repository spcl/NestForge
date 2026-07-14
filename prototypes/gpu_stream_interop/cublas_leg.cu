// External VENDOR-library leg. cuBLAS is a precompiled, closed-source library (libcublas.so / .a) whose
// source we do NOT own -- exactly the "pass a stream to an .a library" case from the design question.
// This file is the thin extern-C shim a nest-forge ExternalCall would wrap around such a library. It
// proves the driver's cudaStream_t can be handed to a real vendor library via cublasSetStream: same
// stream, device pointers only, and this shim does ZERO data movement (the driver owns all of it).
//
//   cublasSetStream(handle, stream)  binds every subsequent cuBLAS call to the driver's stream, so the
//   BLAS work serializes on the same stream as the CUDA and OpenACC legs -- no separate stream, no sync.
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// out-of-place would need a copy; cuBLAS scal is in-place: d_x <- alpha * d_x, on the passed stream.
extern "C" void cublas_scal(double* d_x, long n, double alpha, void* stream) {
    // One handle, created once on first use (C++ guarantees thread-safe function-local static init) and
    // reused. cublasCreate touches the device (allocs workspace), so we keep it off the per-call path. A
    // real ExternalCall env would create it at program init; here it lives for the process, freed at exit.
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        const cublasStatus_t st = cublasCreate(&h);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %d\n", static_cast<int>(st));
        }
        return h;
    }();

    cublasSetStream(handle, static_cast<cudaStream_t>(stream));                     // hand OUR stream to the vendor lib
    const double a = alpha;                                                         // host-pointer mode: alpha on host
    const cublasStatus_t st = cublasDscal(handle, static_cast<int>(n), &a, d_x, 1); // device pointer d_x, stride 1
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasDscal failed: %d\n", static_cast<int>(st));
    }
}
