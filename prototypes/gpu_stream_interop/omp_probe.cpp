// Probe: the OpenMP-target leg is the ONE case that cannot bind to the driver's shared stream. This
// driver shows both the fact and the fix:
//   (A) NO BRIDGE  -- cuda_scale runs on our stream, then omp_add10 runs on OpenMP's own (default) stream
//                     with no synchronization. If OMP ran on a different stream than ours, it reads d
//                     before cuda_scale finishes -> races. We RUN this many times and count mismatches.
//   (B) EVENT BRIDGE -- record a CUDA event on our stream after cuda_scale, make OMP's default stream
//                     wait on it (cudaStreamWaitEvent(nullptr, ev)), then omp_add10, then a matching event
//                     so our stream waits for OMP. This is how the DaCe driver would splice a non-stream-
//                     bindable OMP leg into the shared-stream pipeline WITHOUT a full host sync.
// Correct result = (a*2) + 10. Contrast with driver.cpp, where CUDA/OpenACC/cuBLAS need no bridge at all.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_check.h"

extern "C" void cuda_scale(const double* d_in, double* d_out, long n, void* stream);
extern "C" void omp_add10(double* d_x, long n, void* stream);

int main() {
    const long n = 1L << 22;
    const size_t bytes = n * sizeof(double);
    std::vector<double> h_a(n);
    std::vector<double> h_out(n);
    for (long i = 0; i < n; ++i) {
        h_a[i] = static_cast<double>(i) * 0.5;
    }

    cudaStream_t stream = nullptr;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t ev = nullptr;
    CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

    double* d_a = nullptr;
    double* d = nullptr;
    CK(cudaMalloc(&d_a, bytes));
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    constexpr double tol = 1e-9;
    const int iters = 30;

    // (A) no bridge: OMP on its own stream, no ordering vs our stream.
    long bad_nobridge = 0;
    for (int it = 0; it < iters; ++it) {
        CK(cudaMemsetAsync(d, 0, bytes, stream));
        cuda_scale(d_a, d, n, static_cast<void*>(stream)); // d = a*2 on OUR stream
        omp_add10(d, n, nullptr);                          // d += 10 on OMP's stream -- NO wait for our stream
        CK(cudaMemcpyAsync(h_out.data(), d, bytes, cudaMemcpyDeviceToHost, stream));
        CK(cudaDeviceSynchronize());
        long bad = 0;
        for (long i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (h_a[i] * 2.0 + 10.0)) > tol) {
                ++bad;
            }
        }
        if (bad != 0) {
            ++bad_nobridge;
        }
    }
    printf("(A) OMP no bridge  : %ld/%d runs mismatched -> %s\n", bad_nobridge, iters,
           bad_nobridge != 0 ? "RACES (OMP cannot join our stream directly)" : "no race seen this run");

    // (B) event bridge: our stream -> OMP default stream -> our stream, no host sync in between.
    long bad_bridge = 0;
    for (int it = 0; it < iters; ++it) {
        CK(cudaMemsetAsync(d, 0, bytes, stream));
        cuda_scale(d_a, d, n, static_cast<void*>(stream)); // d = a*2 on OUR stream
        CK(cudaEventRecord(ev, stream));                   // mark: our stream's work up to here
        CK(cudaStreamWaitEvent(nullptr, ev, 0));           // OMP's default stream waits for our stream
        omp_add10(d, n, nullptr);                          // d += 10, now ordered AFTER cuda_scale
        CK(cudaEventRecord(ev, nullptr));                  // mark: OMP's work (on the default stream)
        CK(cudaStreamWaitEvent(stream, ev, 0));            // our stream waits for OMP
        CK(cudaMemcpyAsync(h_out.data(), d, bytes, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        long bad = 0;
        for (long i = 0; i < n; ++i) {
            if (std::fabs(h_out[i] - (h_a[i] * 2.0 + 10.0)) > tol) {
                ++bad;
            }
        }
        if (bad != 0) {
            ++bad_bridge;
        }
    }
    printf("(B) OMP event bridge: %ld/%d runs mismatched -> %s\n", bad_bridge, iters,
           bad_bridge != 0 ? "FAIL" : "PASS (event bridge orders OMP into the shared-stream pipeline)");

    cudaFree(d_a);
    cudaFree(d);
    cudaEventDestroy(ev);
    CK(cudaStreamDestroy(stream));
    return bad_bridge != 0 ? 1 : 0;
}
