// Negative control: same two .a kernels, but acc_add1 runs on a SEPARATE stream from cuda_scale with
// NO cross-stream synchronization. If ordering matters (it does), acc_add1 reads d_mid before/while
// cuda_scale writes it -> races -> mismatches. Contrast with driver.cpp (one shared stream = correct).
// Run many iterations to expose the race. Proves the shared-stream result is due to ordering, not luck.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_check.h"

extern "C" void cuda_scale(const double* d_in, double* d_out, long n, void* stream);
extern "C" void acc_add1(const double* d_in, double* d_out, long n, void* stream);

int main() {
    const long n = 1L << 22; // bigger => wider race window
    const size_t bytes = n * sizeof(double);
    std::vector<double> h_a(n);
    std::vector<double> h_out(n);
    for (long i = 0; i < n; ++i) {
        h_a[i] = static_cast<double>(i) * 0.5;
    }

    cudaStream_t s_cuda = nullptr;
    cudaStream_t s_acc = nullptr;
    CK(cudaStreamCreateWithFlags(&s_cuda, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_acc, cudaStreamNonBlocking));

    double* d_a = nullptr;
    double* d_mid = nullptr;
    double* d_out = nullptr;
    CK(cudaMalloc(&d_a, bytes));
    CK(cudaMalloc(&d_mid, bytes));
    CK(cudaMalloc(&d_out, bytes));
    CK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    constexpr double tol = 1e-12;
    const int iters = 30;
    long total_bad_runs = 0;
    for (int it = 0; it < iters; ++it) {
        CK(cudaMemsetAsync(d_mid, 0, bytes, s_cuda));          // wipe so a stale read is visible
        cuda_scale(d_a, d_mid, n, static_cast<void*>(s_cuda)); // on s_cuda
        acc_add1(d_mid, d_out, n, static_cast<void*>(s_acc));  // on s_acc, NO wait for s_cuda -> race
        CK(cudaMemcpyAsync(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost, s_acc));
        CK(cudaDeviceSynchronize());
        long bad = 0;
        for (long i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (h_a[i] * 2.0 + 1.0)) > tol) {
                ++bad;
            }
        }
        if (bad != 0) {
            ++total_bad_runs;
        }
    }
    printf("NEGATIVE CONTROL (separate streams, no sync): %ld/%d runs had mismatches -> %s\n", total_bad_runs, iters,
           total_bad_runs != 0 ? "RACES as expected (ordering matters)" : "no race observed this run");

    cudaFree(d_a);
    cudaFree(d_mid);
    cudaFree(d_out);
    CK(cudaStreamDestroy(s_cuda));
    CK(cudaStreamDestroy(s_acc));
    return 0;
}
