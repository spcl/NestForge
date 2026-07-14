// External VENDOR-library leg. cuBLAS is a precompiled, closed-source library (libcublas.so / .a) whose
// source we do NOT own -- exactly the "pass a stream to an .a library" case from the design question.
// This file is the thin extern-C shim a nest-forge ExternalCall would wrap around such a library. It
// proves the driver's cudaStream_t can be handed to a real vendor library via cublasSetStream: same
// stream, device pointers only, and this shim does ZERO data movement (the driver owns all of it).
//
//   cublasSetStream(handle, stream)  binds every subsequent cuBLAS call to the driver's stream, so the
//   BLAS work serializes on the same stream as the CUDA and OpenACC legs -- no separate stream, no sync.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

// One handle, created lazily on first use and reused. cublasCreate touches the device (allocs workspace),
// so we keep it out of the per-call path. A real ExternalCall env would create it once at program init.
static cublasHandle_t g_handle = nullptr;

// out-of-place would need a copy; cuBLAS scal is in-place: d_x <- alpha * d_x, on the passed stream.
extern "C" void cublas_scal(double* d_x, long n, double alpha, void* stream) {
    if (!g_handle) {
        cublasStatus_t s = cublasCreate(&g_handle);
        if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed: %d\n", (int)s); return; }
    }
    cublasSetStream(g_handle, (cudaStream_t)stream);   // <-- hand OUR stream to the external vendor lib
    const double a = alpha;                            // host-pointer mode (default): alpha read from host
    cublasStatus_t s = cublasDscal(g_handle, (int)n, &a, d_x, 1);  // device pointer d_x, stride 1
    if (s != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "cublasDscal failed: %d\n", (int)s);
}
