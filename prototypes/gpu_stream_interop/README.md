# gpu_stream_interop — passing one DaCe-created CUDA stream to external, differently-compiled `.a` libs

**Question this answers.** nest-forge extracts a loop-nest, farms it to a compiler, and links the result
back into a DaCe-generated whole-program driver as an `ExternalCall`. On GPU, each extracted nest becomes
a separately-compiled static library that may come from a *different* toolchain (nvcc, nvfortran/OpenACC,
nvc++/OpenMP) or be a *closed-source vendor library* (cuBLAS, cuFFT, cuSOLVER). The DaCe driver owns the
one stream (`__state->gpu_context->streams[i]`, aliased `__dace_current_stream`) and all data movement.
So: **can we pass a stream object we create into each of these external `.a` libraries, and will they all
serialize on it?** Yes — for every mechanism that exposes a stream-set API. This directory proves it on
hardware, end to end, device pointers only, driver-owned movement.

Hardware/toolchain used: RTX 4050 (Ada, `cc89`/`sm_89`), nvhpc SDK 26.3, CUDA 13.3.

## The ABI (fixed by the driver)

Every GPU `.a` entry receives **raw device pointers + the stream**, does zero data movement, and launches
on the passed stream:

```c
extern "C" void <sym>(const double* d_in, double* d_out, long n, void* stream);  /* stream = cudaStream_t */
```

`d_in`/`d_out` are `cudaMalloc`ed by the driver; the kernel must treat them as already-resident and must
NOT copy. `stream` is the driver's `cudaStream_t` as an opaque `void*`.

## The four legs and how each takes the stream

| leg | file | toolchain | device pointers via | stream bound via | joins shared stream directly? |
|-----|------|-----------|---------------------|------------------|-------------------------------|
| CUDA C++      | `cuda_scale.cu` | nvcc            | raw pointer args        | `<<<...,0,(cudaStream_t)stream>>>`      | **yes** |
| OpenACC Fortran | `acc_add1.f90` | nvfortran `-acc` | `deviceptr(...)` + `c_f_pointer` | `acc_set_cuda_stream(q,stream)` + `async(q)` | **yes** |
| cuBLAS (vendor) | `cublas_leg.cu` | nvcc + `-cudalib=cublas` | raw pointer args (cuBLAS takes device ptrs) | `cublasSetStream(handle, stream)` | **yes** |
| OpenMP-target C | `omp_leg.c` | nvc `-mp=gpu` | `is_device_ptr(...)` | *(none — see below)* | **no — needs a CUDA-event bridge** |

**cuBLAS is the headline result**: cuBLAS is a precompiled, closed-source library whose source we do not
own — the exact "pass a stream to an `.a`" case. `cublasSetStream(handle, stream)` binds *every subsequent*
cuBLAS call to our stream. `cublas_leg.cu` is the thin extern-C shim a nest-forge `ExternalCall` would
wrap; the same one-line pattern applies to cuFFT (`cufftSetStream`) and cuSOLVER (`cusolverDnSetStream`).

## Results (from `build.sh`)

`driver.cpp` runs the three stream-bindable legs on ONE stream we create — a three-toolchain pipeline
`d = ((a*2) + 1) * 3`:

```
n=1048576  maxerr=0  mismatches=0  ->  PASS (one shared stream ordered CUDA + OpenACC + cuBLAS-vendor-lib)
```

Bit-exact. The OpenACC kernel read what the CUDA kernel wrote and cuBLAS scaled what OpenACC wrote, so all
three serialized on the single shared stream with no host sync and no per-leg stream of their own.

`omp_probe.cpp` isolates the OpenMP-target leg, the one mechanism with **no** way to run on an externally
supplied stream (OpenMP's `interop` construct runs the other direction — it hands a stream *out* of
OpenMP to a foreign lib, e.g. cuBLAS; there is no portable "run this target region on the caller's
`cudaStream_t`"):

```
(A) OMP no bridge  : 29/30 runs mismatched -> RACES (OMP cannot join our stream directly)
(B) OMP event bridge: 0/30 runs mismatched -> PASS (event bridge orders OMP into the shared-stream pipeline)
```

So an OpenMP-target leg is still usable in the pipeline — the driver splices it in with a **CUDA event**
(`cudaEventRecord(ev, ourStream); cudaStreamWaitEvent(ompStream, ev)` both directions), NOT a full host
sync. `driver_control.cpp` is the analogous negative control for the CUDA+OpenACC pair (separate streams,
no sync → 29/30 race), confirming the shared-stream ordering is real, not luck.

## Constraints this validates (carry into the arena/env model)

- **One vendor backend.** CUDA-C++ + nvfortran-OpenACC + nvhpc-OpenMP + cuBLAS all coexist because they
  share the one NVIDIA CUDA runtime. A HIP `.a` cannot join — it is the mutually-exclusive alt backend
  (compile-time swap).
- **One dynamically-linked libcudart, one primary CUDA context, one device.** DaCe uses the runtime-API
  primary context, which nvfortran/OpenACC/OpenMP and cuBLAS all share transparently. Do NOT static-link a
  second cudart or a second runtime state invalidates the shared handle. (`-cudalib=cublas` links the
  shared cuBLAS; the static `libcublas_static.a` additionally needs `libcublasLt` — prefer shared.)
- **DaCe owns stream lifetime.** Kernels receive the handle; they never create or destroy it.
- **Stream-bindable vs bridge.** CUDA / OpenACC / cuBLAS/cuFFT/cuSOLVER bind the stream directly. OpenMP-
  target does not — bridge it with a CUDA event. Prefer OpenACC for the Fortran GPU path and CUDA/cuBLAS
  for C++; reach for OpenMP-target only when required, and then bridge.

## Build / run

```bash
bash build.sh        # builds the 4 legs + 2 drivers, runs driver (PASS) then omp_probe (A races, B passes)
```

Needs a GPU + nvhpc on PATH (`/opt/nvidia/hpc_sdk/.../26.3/compilers/bin`). Build artifacts (`*.o`,
`*.a`, `*.mod`, `driver`, `driver_control`, `omp_probe`) are git-ignored; only sources are tracked.

## Files

- `cuda_scale.cu` — CUDA C++ leg (`d_out = d_in * 2`), `<<<...,stream>>>`.
- `acc_add1.f90` — OpenACC Fortran leg (`d_out = d_in + 1`), `acc_set_cuda_stream` + `async` + `deviceptr`.
- `cublas_leg.cu` — cuBLAS vendor-lib shim (`d_x *= alpha`), `cublasSetStream` + `cublasDscal`.
- `omp_leg.c` — OpenMP-target C leg (`d_x += 10`), `is_device_ptr`; stream not bindable (bridged by caller).
- `driver.cpp` — three stream-bindable legs on one shared stream → bit-exact PASS.
- `omp_probe.cpp` — OpenMP leg: no-bridge race vs CUDA-event-bridge fix.
- `driver_control.cpp` — negative control: CUDA+OpenACC on separate streams, no sync → races.
- `build.sh` — builds and runs everything.
