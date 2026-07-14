// Negative control: same two .a kernels, but acc_add1 runs on a SEPARATE stream from cuda_scale with
// NO cross-stream synchronization. If ordering matters (it does), acc_add1 reads d_mid before/while
// cuda_scale writes it -> races -> mismatches. Contrast with driver.cpp (one shared stream = correct).
// Run many iterations to expose the race. Proves the shared-stream result is due to ordering, not luck.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void cuda_scale(const double* d_in, double* d_out, long n, void* stream);
extern "C" void acc_add1(const double* d_in, double* d_out, long n, void* stream);
#define CK(x)                                                                                                                                                  \
    do {                                                                                                                                                       \
        cudaError_t e = (x);                                                                                                                                   \
        if (e) {                                                                                                                                               \
            fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e));                                                                                               \
            return 2;                                                                                                                                          \
        }                                                                                                                                                      \
    } while (0)

int main() {
    const long n = 1L << 22; // bigger => wider race window
    const size_t bytes = n * sizeof(double);
    double* h_a = (double*)malloc(bytes);
    double* h_out = (double*)malloc(bytes);
    for (long i = 0; i < n; ++i)
        h_a[i] = (double)i * 0.5;

    cudaStream_t s_cuda, s_acc;
    CK(cudaStreamCreateWithFlags(&s_cuda, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_acc, cudaStreamNonBlocking));
    double *d_a, *d_mid, *d_out;
    CK(cudaMalloc(&d_a, bytes));
    CK(cudaMalloc(&d_mid, bytes));
    CK(cudaMalloc(&d_out, bytes));
    CK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    long total_bad_runs = 0;
    const int ITERS = 30;
    for (int it = 0; it < ITERS; ++it) {
        CK(cudaMemsetAsync(d_mid, 0, bytes, s_cuda)); // wipe so a stale read is visible
        cuda_scale(d_a, d_mid, n, (void*)s_cuda);     // on s_cuda
        acc_add1(d_mid, d_out, n, (void*)s_acc);      // on s_acc, NO wait for s_cuda -> race
        CK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, s_acc));
        CK(cudaDeviceSynchronize());
        long bad = 0;
        for (long i = 0; i < n; ++i)
            if (fabs(h_out[i] - (h_a[i] * 2.0 + 1.0)) > 1e-12)
                ++bad;
        if (bad)
            ++total_bad_runs;
    }
    printf("NEGATIVE CONTROL (separate streams, no sync): %d/%d runs had mismatches -> %s\n", (int)total_bad_runs, ITERS,
           total_bad_runs ? "RACES as expected (ordering matters)" : "no race observed this run");
    cudaFree(d_a);
    cudaFree(d_mid);
    cudaFree(d_out);
    free(h_a);
    free(h_out);
    return 0;
}
