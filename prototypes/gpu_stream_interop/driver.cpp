// DaCe-style C++ driver: owns the data AND the stream, farms three nest-kernels onto the SAME stream,
// passing device pointers only. Mirrors what a nest-forge ExternalCall GPU path does: the driver hands
// __state->gpu_context->streams[i] (or __dace_current_stream) to each .a. The three legs are three
// DIFFERENT toolchains sharing one stream WE created:
//   1. CUDA C++ .a        (libcuda_scale) : d_mid = d_a * 2      -- our own __global__, <<<...,stream>>>
//   2. OpenACC Fortran .a (libacc_add1)   : d_b   = d_mid + 1    -- acc_set_cuda_stream + async(q)
//   3. cuBLAS vendor lib  (libcublas_leg) : d_b  *= 3            -- EXTERNAL precompiled lib, cublasSetStream
//
// Leg 3 is the point of this driver: we CREATE the stream and pass it into a closed-source vendor library
// (cuBLAS) via cublasSetStream, so the BLAS call serializes on the same stream as our own kernels.
// Correct out = (a*2 + 1) * 3 proves all three toolchains ordered on the one shared stream.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void cuda_scale(const double* d_in, double* d_out, long n, void* stream);
extern "C" void acc_add1(const double* d_in, double* d_out, long n, void* stream);
extern "C" void cublas_scal(double* d_x, long n, double alpha, void* stream);

#define CK(x)                                                                                                                                                  \
    do {                                                                                                                                                       \
        cudaError_t e = (x);                                                                                                                                   \
        if (e) {                                                                                                                                               \
            fprintf(stderr, "CUDA %s: %s\n", #x, cudaGetErrorString(e));                                                                                       \
            return 2;                                                                                                                                          \
        }                                                                                                                                                      \
    } while (0)

int main() {
    const long n = 1L << 20;
    const size_t bytes = n * sizeof(double);
    double* h_a = (double*)malloc(bytes);
    double* h_out = (double*)malloc(bytes);
    for (long i = 0; i < n; ++i)
        h_a[i] = (double)i * 0.5;

    // One stream, created the way DaCe creates internal_streams[i] -- this is the object we pass around.
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    double *d_a, *d_mid, *d_b;
    CK(cudaMalloc(&d_a, bytes));
    CK(cudaMalloc(&d_mid, bytes));
    CK(cudaMalloc(&d_b, bytes));

    CK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream)); // driver owns movement

    cuda_scale(d_a, d_mid, n, (void*)stream); // CUDA C++ .a  : d_mid = d_a * 2
    acc_add1(d_mid, d_b, n, (void*)stream);   // OpenACC F .a : d_b   = d_mid + 1  (SAME stream)
    cublas_scal(d_b, n, 3.0, (void*)stream);  // cuBLAS vendor: d_b  *= 3           (SAME stream)

    CK(cudaMemcpyAsync(h_out, d_b, bytes, cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));

    long bad = 0;
    double maxerr = 0.0;
    for (long i = 0; i < n; ++i) {
        double want = (h_a[i] * 2.0 + 1.0) * 3.0;
        double err = fabs(h_out[i] - want);
        if (err > 1e-9) {
            if (bad < 5)
                fprintf(stderr, "MISMATCH i=%ld got=%.17g want=%.17g\n", i, h_out[i], want);
            ++bad;
        }
        if (err > maxerr)
            maxerr = err;
    }
    printf("n=%ld  maxerr=%.3g  mismatches=%ld  ->  %s\n", n, maxerr, bad,
           bad == 0 ? "PASS (one shared stream ordered CUDA + OpenACC + cuBLAS-vendor-lib)" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_mid);
    cudaFree(d_b);
    CK(cudaStreamDestroy(stream));
    free(h_a);
    free(h_out);
    return bad == 0 ? 0 : 1;
}
