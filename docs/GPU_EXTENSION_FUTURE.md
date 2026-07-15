> **STATUS: VERY-FUTURE TODO — NOT IMPLEMENTED.** Design/research only, captured 2026-07-11 from a
> multi-agent survey. No GPU code path exists in nest-forge yet. The user's steer: DaCe owns the
> host↔device transfers and a loopnest's data is already resident on-device, so nest-forge would only
> need to emit *naive* device kernels (SYCL / OpenACC / OpenMP-target) per map — no transfer
> management. Revisit after the CPU cross-compiler/cross-language + FP-precision-matrix arena lands.

# Non-tile GPU backends for the nest-forge arena

A proposal to extend the nest-forge arena from CPU-only (`gcc`/`clang` × FP-mode) to GPU code generation for extracted parallel map-nests, using only the locally installed toolchains (nvhpc 26.3, oneAPI 2026.1, gcc 15, clang 21) and the RTX 4050 (sm_89) that is physically present.

The headline conclusion up front: **GPU does not belong in the compiler × FP-flag arena matrix**. It belongs in a separate, DaCe-native GPU-correctness lane. The rest of this document grounds that claim.

---

## 1. What DaCe emits today

DaCe has exactly **one** GPU code generator, and it emits **both** CUDA and HIP from the same class.

- **CUDACodeGen is the sole GPU target.** `dace/codegen/targets/cuda.py:70-73` (`class CUDACodeGen`, `target_name='cuda'`). The `codegen/targets/` directory contains only `cpu.py`, `cuda.py`, `mpi.py`, `snitch.py`, `sve/`, `mlir/` — `cuda.py` is the only GPU emitter. A grep for `sycl|opencl|openacc|omp target|metal` across `dace/codegen` returns only a stray comment in `prettycode.py:85`. `dtypes.Language.OpenCL` exists (`dtypes.py:147`) as a tasklet-language enum but is referenced by no code generator. **There is no SYCL, OpenCL, Metal, OpenACC, or OpenMP-`target` offload path in DaCe.**

- **CUDA and HIP are the same code, parametrized by one string.** The class comment is literally "GPU (CUDA/HIP) code generator" (`cuda.py:71`). A single `self.backend` (`'cuda'` or `'hip'`) is interpolated into every emitted runtime call: `{backend}LaunchKernel`, `{backend}Malloc`, `{backend}DeviceSynchronize`. With `backend=='cuda'` it emits `.cu` files (`self.language='cu'`); with `backend=='hip'` it emits `.cpp` and includes `hip/hip_runtime.h` (`cuda.py:167-168, 383-388`). The backend is selected by config `compiler.cuda.backend` (default `'auto'`), resolved by the detection cascade in `dace/codegen/common.py:117-161` (`nvidia-smi`→cuda, `rocm-smi`→hip, then CMake find-package, env vars, runtime libs). It is a plain string — there is no `Backend` enum.

- **The lowering is one-map-to-one-kernel, non-tile.** CUDACodeGen registers as the map dispatcher for all `GPU_SCHEDULES` (`cuda.py:122`). One `ScheduleType.GPU_Device` map becomes one `__global__ void <name>(...)` kernel (`cuda.py:1776`) plus a `__dace_runkernel_<name>` host wrapper (`cuda.py:1800`) that issues `{backend}LaunchKernel(dim3(grid), dim3(block), ...)` on a stream, guarded by an empty-grid check (`cuda.py:1892-1905`). The map's iteration parameters are materialized **inside** the kernel from `blockIdx.{x,y,z}*blockDim + threadIdx.{x,y,z}` with delinearization for >3 dims (`cuda.py:2342-2377`). This is **raw thread/block index arithmetic — not a Triton/cuTile tile abstraction.**

So: **today DaCe emits CUDA (`.cu`, default when an NVIDIA GPU is seen) and HIP (`.cpp`) from one `CUDACodeGen`, mapping each `GPU_Device` map to one `__global__` kernel + one `{backend}LaunchKernel`.** nest-forge currently uses none of this — `arena.py` and `translate.py` go through the numpy→C translator and host `ctypes`, never through DaCe's GPU codegen.

---

## 2. Candidate non-tile GPU backends

Local hardware/toolchain facts that drive the table: RTX 4050 (sm_89, `/dev/nvidia0` live, driver CUDA 13.2) **is present and runs**; nvhpc 26.3 ships `nvcc 13.1` at `/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1`; oneAPI 2026.1 ships `icpx` (spir64/CPU works, NVIDIA/AMD SYCL targets need un-installed Codeplay plugins); gcc 15 was built with `nvptx-none:amdgcn-amdhsa` offload but the `gcc-15-offload-nvptx` accelerator package is **not** installed; clang 21 ships **no** `libomptarget` device RTL and no default CUDA path; there is **no ROCm/hipcc** anywhere and the only AMD device is an unsupported gfx1103 iGPU.

| Backend | Emittable from a map | Local toolchain ready | Effort | Payoff | How validated | Recommend |
|---|---|---|---|---|---|---|
| **CUDA C** (nvcc 13.1 / clang-21 `-x cuda` / nvc++) | Yes — reuse DaCe `CUDACodeGen`, zero new emitter | **Yes** — RTX 4050 sm_89 + nvcc 13.1 compile *and run* locally, no extra SDK | High | Medium | On-device: cudaMalloc→H2D same oracle inputs→launch→sync→D2H→compare vs numpy. Bit-exact only with `--fmad=false`; else `np.allclose` per `MODE_ATOL` | **Yes — but as a separate lane, not an FP-flag column** |
| **HIP C** (hipcc / amdclang) | Yes — same `CUDACodeGen` with `backend='hip'` | **No** — no ROCm/hipcc/`hip_runtime.h`; gfx1103 iGPU unsupported by ROCm | High | Low | Same shape as CUDA (hipMemcpy D2H), but nothing local to compile or run | No |
| **SYCL** (icpx `-fsycl` / AdaptiveCpp) | Yes, but needs a **new** `numpyto_sycl` emitter (DaCe has no SYCL target) | **CPU only** — icpx spir64 ran a `parallel_for` bit-exact on the Ryzen; no confirmed SYCL GPU device | Medium | Low | ctypes `.so` via `call_native` + buffer/host-accessor or USM D2H copy; CPU-SYCL plausibly bit-exact | No |
| **OpenMP target** (`nvc -mp=gpu` / clang / gcc nvptx) | Yes — `#pragma omp target teams distribute parallel for`, pure C, fits the C-source seam | **Partial** — only `nvc -mp=gpu` out of the box; clang lacks the nvptx `libomptarget`, gcc lacks the installed nvptx accel package | Medium | Low | Offload on RTX 4050, `map(from:/tofrom:)` D2H, compare vs oracle; FP reassociation breaks bit-exact unless `-Mnofma`/`-ffp-contract=off`, reductions still need tolerance | No (thin column only) |
| **OpenACC** (`nvc -acc=gpu` / nvfortran / gcc) | Yes — `#pragma acc parallel loop`, directive, Fortran-friendly | **Partial** — nvc/nvfortran `-acc=gpu` production; gcc 15 nvptx OpenACC weak/WIP | Medium | Low | Same as OpenMP-target (`acc data`/`copyout` D2H); redundant with the CUDA path DaCe already emits | No |
| **stdpar** (`nvc++ -stdpar=gpu`) | Yes — `std::for_each(par_unseq, counting_range, …)`, no language extension | **Yes (nvc++)** — but relies on CUDA Unified Memory for all data movement | Low (source) / Medium (UM) | Low | UM makes host arrays device-visible; run, sync, compare vs oracle; FP reassoc caveats as above | No |
| **Kokkos** (`kokkos` + CUDA/HIP backend) | Yes in principle (`Kokkos::parallel_for` over a range) | **No** — not installed; it is a portability *framework* (new multi-GB dependency), not a compiler on PATH | High | Low | Would run through its CUDA backend to the same RTX 4050; adds a build system, not a lowering strategy | No |

**Why every arena-column framing fails (grounded).** The arena passes host numpy buffers by pointer through `ctypes` (`arena.py:174-197`, `call_native`); a GPU cell needs a whole new device execution path (alloc/H2D/launch/D2H/free) — a new backend, not a new entry in `_CANDIDATE_COMPILERS` (`arena.py:39`). The FP flags are host-compiler flags (`_BASE = -O3 -march=native`, `FP_MODES`, `arena.py:28-36`) that do not transfer to nvcc (which needs `-arch=sm_89`, `--fmad=false`/`--use_fast_math`, CUDA-event timing, and H2D/D2H accounting). Ranking nvcc in the same column as gcc/clang misleads the compiler × flag sweep. And the 6 GB laptop GPU makes any perf ranking non-representative — the real value is **GPU-codegen correctness**, not winning a perf arena.

---

## 3. Ranked shortlist (what to add first)

**#1 — CUDA C as a decoupled GPU-correctness lane, reusing DaCe's `CUDACodeGen`.**
It is the only backend that (a) is genuinely runnable on-device locally (RTX 4050 sm_89 + nvcc 13.1, no extra SDK), and (b) requires **zero new emitter code** — DaCe already lowers a `GPU_Device` map to `.cu` + a `__dace_runkernel` host wrapper (`cuda.py:1776, 1800`). It is also the direct comparison point for DaCe's own only GPU emitter. Add it **not** as an FP-flag column but as a separate lane: set the extracted map's schedule to `GPU_Device`, emit `.cu` via DaCe, nvcc it, run, D2H-compare vs the numpy oracle. This validates the device-scope cut (§4/M4) at the lowest possible cost.

**#2 — OpenMP-target via `nvc -mp=gpu`, as one thin directive column (optional).**
Rationale: it is the single GPU model that emits **pure C source with no kernel syntax** (`#pragma omp target teams distribute parallel for collapse(k)`), so it maps most cleanly onto nest-forge's existing `translate → C → compile` seam (`translate.py:emit_sources(target="c")`), and `nvc -mp=gpu` works out of the box. Gate it hard: only `nvc` is locally ready (clang has no nvptx `libomptarget`, gcc's nvptx accel package is not installed), and GPU FMA/reduction reassociation breaks bit-exactness — `MODE_ATOL["ieee-strict"]==0.0` (`arena.py:37`) is unreachable on-device for reductions even with `-Mnofma`. Worth it only if you want a directive-vs-CUDA accuracy contrast; otherwise skip.

**#3 — stdpar (`nvc++ -stdpar=gpu`), as a stretch.**
Cheapest source shape (`std::for_each(par_unseq, …)`, ISO C++ only) and nvc++ is installed, but it depends on CUDA Unified Memory for all data movement, which changes the buffer model versus the explicit-copy CUDA lane. Defer until #1 is solid.

**Deferred / rejected:** HIP (no ROCm; gfx1103 unsupported), SYCL-GPU (no confirmed device; CPU-SYCL duplicates the existing CPU columns and needs a net-new emitter), OpenACC-as-its-own-column (redundant with CUDA, which DaCe already emits), Kokkos (a framework/dependency, not a PATH compiler).

---

## 4. The concrete integration seam

### 4.1 map → kernel: reuse DaCe codegen, do not hand-write a `.cu` emitter

For the CUDA lane, the emit step **bypasses the numpy→C translator** (`translate.py`, `emit_numpy.py`) and instead drives DaCe's own GPU codegen on the extracted standalone SDFG:

1. On `boundary.standalone_sdfg`, set the extracted map's `schedule = ScheduleType.GPU_Device` (and mark its arrays `GPU_Global`) — the *device-scope cut*. This is the exact GPU analogue of the CPU schedule cut already specified in `PARALLEL.md §2`: "a schedule-domain cut, the same shape as the host-wrapper / GPU-device cut." The choice of *which* map level becomes the device boundary is the same ancestor-walk decision that computes `parent_is_parallel` (`PARALLEL.md §1-2`).
2. Call DaCe codegen (the path `build.py:build_sdfg` already uses for the CPU program) → get `.cu` + the `__dace_runkernel_<name>` host wrapper (`cuda.py:1776, 1800-1905`) compiled into a `.so`.
3. Because DaCe's `CUDACodeGen` already lowers WCR reductions with atomics/tree-reduction, the embarrassingly-parallel-only soundness hazard of a naive hand emitter is avoided — the reduction subset (`tests/test_wcr_emit.py`) is handled by reuse, not reimplementation. This is a decisive reason to reuse rather than emit by hand.

### 4.2 build.py / arena: a **sibling** GPU matrix, not a folded-in column

Add a GPU compiler axis parallel to `_CANDIDATE_COMPILERS` (`arena.py:39`), e.g. `{"nvcc": <nvhpc nvcc>, "clang-cuda": <clang-21 -x cuda --cuda-path=/opt/nvidia/hpc_sdk/.../cuda/13.1>}`, with a **GPU-specific flag map** that replaces `FP_MODES` (host flags do not transfer):

- `ieee-strict` → `-arch=sm_89 --fmad=false` (best-effort; note reductions still diverge),
- `fast-but-ieee` → `-arch=sm_89` (FMA on),
- `fast-math` → `-arch=sm_89 --use_fast_math`.

Timing is CUDA-event based inside the child, and the report accounts H2D/D2H separately from kernel time. Reuse `MODE_ATOL` (`arena.py:37`) for the compare tolerance, but document that GPU `ieee-strict` is a floor, not `0.0`. This produces `GpuCell` records alongside the CPU `Cell`s (`arena.py:149-159`), reported in their own section — never interleaved with the gcc/clang × FP grid.

### 4.3 How a GPU cell validates vs the numpy oracle

The same oracle (`run_oracle`, `arena.py:133`) and the same `make_inputs` arrays (`arena.py:118`) feed both lanes, so the comparison is apples-to-apples on inputs. A new `call_native_gpu` (sibling of `call_native`, `arena.py:174`) runs **inside the `run_isolated` fork** (`isolation.py`; the standing rule "always fork compiled kernels" applies — a device fault kills only the child):

1. `cudaMalloc` a device buffer per manifest `input_args`, `cudaMemcpy` H2D the oracle inputs,
2. invoke the DaCe `.so`'s `__dace_runkernel` entry via `ctypes` (device pointers in its `__state`),
3. `cudaDeviceSynchronize`, `cudaMemcpy` D2H the `boundary.outputs`,
4. `maxdiff` vs the oracle (`arena.py:200`) under `MODE_ATOL`, free buffers.

Only the small `{ok, maxdiff, time_us}` summary crosses the fork pipe, exactly as the CPU path does today (`arena.py:253-256`).

### 4.4 Mapping to the plan: M4 (device-scope cut) + `ExpandExternCallGPU`

- **M4 = the device-scope-cut strategy.** nest-forge already has `outer` / `skip-taskloops` / `innermost` extraction strategies (`README.md` M1, `strategies.py`). M4 adds a `device` strategy that picks the map level to mark `GPU_Device` — the host/device boundary — the direct sibling of `parent_is_parallel` in `PARALLEL.md §2` (one level owns the parallelism; everything below is per-thread/per-lane). `PARALLEL.md §5`'s order-preserving vs order-changing FP taxonomy carries over unchanged: a GPU reduction is a reassociation event, gated against the sequential ieee-strict baseline by `fp_risk`.
- **`ExpandExternCallGPU` = the GPU sibling of `ExpandExternCall`.** The whole-program libnode is `ExternalCall` with two implementations, `DaceReference` (rebuild-as-NestedSDFG fallback/oracle) and `ExternCall` (link the winning `.so`), in `libnode.py:96-134`. Add a third implementation `ExpandExternCallGPU` that links the DaCe-built `.cu` `.so` **plus its `__dace_runkernel` host wrapper** into the whole-program SDFG — structurally identical to how `ExpandExternCall` (`libnode.py:108-126`) wraps the CPU `.so` in a CPP tasklet, but the wrapper it calls does the H2D/launch/D2H. `DaceReference` (`libnode.py:96`) remains the correctness oracle for the GPU cell just as it is for CPU (`README.md:22`).

---

## 5. Tile-based models — explicitly out of scope

The survey's non-tile framing is deliberate. Out of scope, and why:

- **Triton** (tile / block-pointer DSL), **NVIDIA cuTile / CUTLASS CuTe** tile abstractions, and any **warp-tile MMA / `wmma` fragment** API. These are a fundamentally different lowering — an affine tile IR, not an affine-map→SPMD-thread mapping. The survey contrasts DaCe's emitter as "raw thread/block index arithmetic, not a Triton/cuTile tile abstraction" (`cuda.py` evidence). A map→kernel arena does not target them.
- **SYCL `nd_range` with explicit work-group tiling** is out; only the plain `sycl::range<N>` form (no tiling) is in scope for a bare affine map.
- Note the in-scope subtlety: DaCe's `GPU_ThreadBlock` / `GPU_ThreadBlock_Dynamic` schedules *look* tile-shaped but are still emitted through the same non-tile raw-index CUDA path (`cuda.py`). nest-forge marks only `GPU_Device`, so this ambiguity does not arise — we never request the thread-block tiling schedules.

---

**Bottom line.** DaCe emits CUDA and HIP today from one `CUDACodeGen` (`cuda.py:70-73`), non-tile, one `GPU_Device` map → one `__global__` kernel + one `{backend}LaunchKernel`. The only backend worth adding first is **CUDA C**, and only as a **DaCe-native GPU-correctness lane** (device-scope-cut M4 + `ExpandExternCallGPU`), decoupled from the CPU compiler × FP-flag arena — because a GPU cell is a new device execution path, its flags and timing don't share the host axis, and a 6 GB laptop GPU makes perf ranking meaningless while codegen correctness is the real prize.

---

## 6. Deferred GPU items carried over from the CPU arena (Phase 2/2c/1.3)

Three concrete items surfaced while landing the CPU polyhedral lanes and the tile-op vectorization axis. They are GPU by nature, so the CPU-first decision parks them here rather than in the arena.

### 6.1 Externalize-before-offload — a hard ordering invariant

If DaCe offloads a loop nest to the device and then a polyhedral tool (PPCG below) independently decides the *same* nest is GPU-viable, the offload decision has been made twice, inconsistently. So the order is fixed and non-negotiable: **externalize each loop nest into a library call FIRST** (`pass_lower.lower_nests_to_external_call` → the `ExternalCall` libnode), *then* each tool decides on its own whether its kernel is offloadable. No lane may pre-decide offload before extraction. This is CPU-irrelevant today (every CPU lane compiles a host `.so`), which is exactly why it is recorded now — it only starts to bite once two independent GPU emitters (DaCe's `CUDACodeGen` and PPCG) can both claim the same nest.

### 6.2 PPCG — polyhedral C → CUDA (the GPU sibling of the Pluto lane)

PPCG is a **standalone** polyhedral source-to-source tool (not a numpyto target and not a compiler flag) that consumes the SAME affine `#pragma scop` C input as Pluto's `polycc` — nest-forge already emits it as `<base>_pluto_input.c` — and transforms it into **CUDA**. So a PPCG lane is the GPU analogue of the CPU Pluto lane (`nestforge/perf/pluto_lane.py`): probe `ppcg` on PATH; skip-with-reason when absent (it is a distinct polyhedral toolchain, off every box that lacks the container); gate the scop through the shared affine-index detector (`optarena.pluto_affine.scop_nonaffine_reason`) exactly as the Pluto lane does; then `ppcg <scop> → .cu`, `nvcc -arch=sm_89`, run on-device, D2H-compare vs the numpy oracle (§4.3). Because its output is CUDA, it belongs in the GPU-correctness lane (§3/§4), never the CPU compiler × FP-flag grid.

**Install.** PPCG has no spack package and is source-build-only (isl + pet + libclang + gmp + libyaml). It does NOT warrant a fresh recipe: optarena's `containers/pluto.Dockerfile` already pins clang-17/llvm-17-dev + libclang-17-dev, autoconf/automake/libtool/pkg-config, libgmp-dev, libyaml-dev, and builds isl + pet for `polycc` — every PPCG dependency. Adding `ppcg` there is a few `RUN` lines that reuse the whole pinned tree and the CI job that already builds+verifies the image; no new `.so`, no new pinned toolchain. That container is the install path when the GPU work begins — it is intentionally NOT built now, since a CUDA-emitting lane cannot land under the CPU-first scope.

### 6.3 GPU tile-op vectorization presets + the half2 rule

The CPU tile-op vectorizer (`VectorizeConfig`, `VectorizeCPUMultiDim`) has a `device` knob (`config.py:59`, `DeviceType.CPU`/`GPU`) and a `BRANCHED_TAIL` GPU remainder strategy that the CPU arena never exercises. The GPU presets to sweep once a GPU lane exists: `gpu-scalar`, `gpu-fp16-scalar`, `gpu-w2-branched[-fma]`, `gpu-w4-even[-fma]` — the GPU analogue of the staged CPU screening in `vectorize_variants.py`, with `BRANCHED_TAIL` replacing the CPU `MASKED_TAIL`/`SCALAR_POSTAMBLE` remainders (a warp-divergent branch is the GPU-natural tail, not a mask).

**The half2 rule (record it before it is lost):** on fp16, two lanes pack into one `half2` register, so a width-2 fp16 kernel is the fast path — BUT a **masked** remainder defeats `half2` packing (you cannot half-predicate a packed pair), so the emitter must fall back to scalar whenever the tail is masked. Concretely: fp16 → `half2` only when the tail is NOT masked (`!Masked`); otherwise scalar. This is why the GPU presets pair `w2` with `branched` (a branch, not a mask) and why `assume_even` (no remainder at all) is the cleanest fp16 path. The CPU `fp16→half2 else scalar` rule already lives in the vectorizer; the GPU remainder interaction is the piece that must not be re-derived from scratch when the GPU lane lands.

Grounding files (all absolute): `/home/primrose/Work/nest-forge/nestforge/arena.py`, `.../nestforge/build.py`, `.../nestforge/libnode.py`, `.../nestforge/translate.py`, `.../nestforge/emit_numpy.py`, `.../PARALLEL.md`, `.../README.md`; DaCe `dace/codegen/targets/cuda.py`, `dace/codegen/common.py`, `dace/dtypes.py`.