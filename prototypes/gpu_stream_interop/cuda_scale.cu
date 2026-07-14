// CUDA C++ kernel, compiled to libcuda_scale.a (mimics a DaCe-generated / nvcc ExternalCall variant).
// Device-pointer ABI: d_in/d_out are ALREADY on the GPU; the caller (driver) owns all data movement.
// The stream is handed in by the caller and the launch runs on it.
#include <cuda_runtime.h>

__global__ void scale_kernel(const double* in, double* out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0;
}

extern "C" void cuda_scale(const double* d_in, double* d_out, long n, void* stream) {
    const int threads = 256;
    long blocks = (n + threads - 1) / threads;
    scale_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(d_in, d_out, n);
}
