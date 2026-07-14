// OpenMP-target C leg (nvc -mp=gpu). is_device_ptr is the OpenMP analogue of OpenACC deviceptr and the
// CUDA raw-pointer contract: d_x is ALREADY resident device memory -- use it directly, no map/copy.
// This proves the DEVICE-POINTER half of the ABI for OpenMP.
//
// The STREAM half is the catch, and the reason OpenMP is the odd one out. CUDA (<<<...,stream>>>),
// OpenACC (acc_set_cuda_stream + async), and cuBLAS (cublasSetStream) all expose a way to run ON an
// externally supplied cudaStream_t. OpenMP-target does NOT: the standard `interop` construct runs the
// other direction (it hands a stream OUT of OpenMP to a foreign library, e.g. cuBLAS), and there is no
// portable "run this target region on the caller's cudaStream_t". So an OpenMP-target leg cannot directly
// join the driver's shared stream; the driver must bridge ordering with a CUDA event / host sync. The
// `stream` argument is accepted for a uniform ABI but is deliberately unused here -- see omp_probe.cpp.
#include <stddef.h>

void omp_add10(double* d_x, long n, void* stream) {
    (void)stream; // OMP-target cannot bind to an external stream; ordering is bridged by the caller
#pragma omp target teams distribute parallel for is_device_ptr(d_x)
    for (long i = 0; i < n; ++i)
        d_x[i] += 10.0;
}
