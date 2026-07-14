#!/usr/bin/env bash
# Build 3 separately-compiled static libs and one DaCe-style C++ driver that shares ONE cudaStream_t we
# create across all of them: a CUDA C++ kernel, an OpenACC Fortran kernel, and an EXTERNAL vendor library
# (cuBLAS) reached through a thin extern-C shim. RTX 4050 = Ada = cc89 / sm_89.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH
CC=89

rm -f *.o *.a driver driver_control

echo "[1] CUDA C++ kernel -> libcuda_scale.a (nvcc)"
nvcc -O2 -arch=sm_${CC} -c cuda_scale.cu -o cuda_scale.o
ar rcs libcuda_scale.a cuda_scale.o

echo "[2] OpenACC Fortran kernel -> libacc_add1.a (nvfortran -acc)"
nvfortran -O2 -acc -gpu=cc${CC} -c acc_add1.f90 -o acc_add1.o
ar rcs libacc_add1.a acc_add1.o

echo "[3] cuBLAS vendor-lib shim -> libcublas_leg.a (nvcc; the shim calls cublasSetStream + cublasDscal)"
nvcc -O2 -arch=sm_${CC} -c cublas_leg.cu -o cublas_leg.o
ar rcs libcublas_leg.a cublas_leg.o

echo "[4] Driver + link (nvc++ -cuda -acc pull cudart + OpenACC rt; -cudalib=cublas pulls the vendor lib)"
nvc++ -O2 -cuda -acc -gpu=cc${CC} driver.cpp -L. \
      -lcuda_scale -lacc_add1 -lcublas_leg -cudalib=cublas -fortranlibs -o driver

echo "[5] Run"
./driver

# ---- OpenMP-target leg: the odd one out (cannot bind an external stream). Built separately so its
#      event-bridge probe stays independent of the proven CUDA+OpenACC+cuBLAS shared-stream pipeline. ----
echo "[6] OpenMP-target C leg -> libomp_leg.a (nvc -mp=gpu)"
nvc -O2 -mp=gpu -gpu=cc${CC} -c omp_leg.c -o omp_leg.o
ar rcs libomp_leg.a omp_leg.o

echo "[7] OMP probe + link (nvc++ -mp=gpu -cuda) : shows no-bridge race vs event-bridge fix"
nvc++ -O2 -mp=gpu -cuda -gpu=cc${CC} omp_probe.cpp -L. -lcuda_scale -lomp_leg -o omp_probe

echo "[8] Run OMP probe"
./omp_probe
