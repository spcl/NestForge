# What NestForge needs from HPCAgent-Bench

[Designs](README.md)

Written 2026-11-16 for the bench maintainers, against HPCAgent-Bench `cc1124254` (on the way to v1.0) and NestForge
`bf068ae`. Each request says what is missing, why NestForge needs it, and a proposed design. NestForge patches none
of them on its side. B1 to B5, B11, and B12 belong to the distributed work ([comm-nodes.md](comm-nodes.md)), B6
to B10 are independent of it, and B13 and B14 are baselines for the study.

| # | Request | Blocks | Size |
|---|---|---|---|
| B1 | The canonical form renders the distributed entry | distributed finalize | medium |
| B2 | `/profile` runs a library at given rank counts | distributed `get_performance` | medium |
| B3 | A DaCe form of each distributed reference | every distributed column | small per kernel |
| B4 | Tuner episodes on a distributed task | every distributed column | medium |
| B5 | Layout requests on a distributed array | `relayout` on distributed arguments | design question |
| B6 | Precision per argument in the entry | `relayout` casts | medium |
| B7 | A tool call's route request names the assistant message | token split per phase | small |
| B8 | A tuner flavor can decline a kernel as unsupported | flavors that cannot take a residency (B4) | small |
| B9 | A route request names the episode's packet | nothing; hardening | small |
| B10 | mypy error in `config.py` | clean hooks | open as bench PR #190 |
| B11 | A NestForge submission adds no communication | the claim that the agent never writes MPI | medium |
| B12 | Two application tasks: the ICON dynamical core and OMEN | the application experiment | large |
| B13 | An LLVM-pass agent packet | the catalogue baseline of the NestForge study | medium |
| B14 | A GEAK harness plugin | a GPU baseline from the literature | open, decide first |

## B1: the canonical form renders the distributed entry

**Missing.** `cpf.bridge.render_canonical(spec, name, sdfg, language, precision, target, hints)` renders the
single-node entry for `Target.CPU` and `Target.GPU`. The distributed ABI (abi_contract.md, section 12) has no
rendering: `<base>_mpi(local tiles, local scalars, MPI_Fint comm, workspace, workspace_size)`.

**Proposed design.**

1. `Target` gains `DISTRIBUTED`. The rendered entry is `<base>_mpi`, with the arguments in the contract's order.
2. A size symbol on a distributed axis is bound to the rank's local extent, as the contract says. The SDFG must not
   size a replicated array by such a symbol, and the renderer refuses one that does, naming the array.
3. The SDFG takes the communicator as a scalar argument named `comm`, the Fortran handle. DaCe's `CommF2c` node
   converts it inside the program. The renderer refuses an SDFG whose communication nodes use the world communicator.
4. The form's `link` carries the MPI link flags of the image, and `languages.build_shared_lib_commands` accepts the
   MPI compile flags, so a plugin builds the library exactly as the judge would.
5. A test renders `jacobi_2d` from its DaCe form (B3) and grades it at 1, 2, and 4 ranks.

## B2: run a library at given rank counts from a plugin

**Missing.** The judge's `/profile` route times a single-node library, and it takes no rank count. A plugin has no
call that runs a library under the distributed driver (`support/bindings/mpi_driver.py`) and returns the time and
the verdict without recording a grade.

**Proposed design.** `/profile` gains an optional field `ranks: list[int]`, valid on a distributed setup only, and
its answer gains one entry per rank count:

```python
class DistributedRun(pydantic.BaseModel):    # frozen
    ranks: int
    time_us: float | None        # the timed bracket, slowest rank (mpi-wtime-max)
    mpi_time_us: float | None    # time inside MPI calls on the slowest rank
    correct: bool | None         # the gathered output against the single-node reference
    mismatched: list[str]
    error: str
```

- The run uses the episode's preset, datatype, launcher, and distribution, as a `/submit` grade would, and it is
  never recorded as a grade.
- `ranks` must be a subset of the setup's `mpi.rank_counts`. Another count is answered 400.
- `mpi_time_us` comes from a PMPI shim linked into the driver. It sums the `MPI_Wtime` differences around every MPI
  call of the kernel and reduces the sum with `MPI_MAX`. The shim costs two timer reads per call and is on for this
  route only, never for a grade. A device kernel that uses NCCL or RCCL reports `None`.
- The call counts against the episode's profile budget, as every `/profile` call does.

## B3: a DaCe form of each distributed reference

**Missing.** The eight distributed references (`scaled_add_mpi`, `mat_scaled_add_mpi`, `jacobi_2d_mpi`,
`heat_3d_mpi`, each as `.c` and `.py`) are C and plain `mpi4py`. No DaCe program exists for a distributed kernel, so
no `dace_*` column and no plugin on top of one can take a distributed task.

**Proposed design.** One `<kernel>_dace_mpi.py` per kernel beside the references: a `@dace.program` over the local
tiles that takes `comm` and uses `mpi4py` calls DaCe's frontend turns into communication nodes (`comm.Isend`,
`comm.Irecv`, `MPI.Request.Waitall`, `comm.Allreduce`). The halo fixtures use nonblocking exchanges, so a move can
overlap them with compute. `DaceFramework.build_sdfgs` loads this file for a distributed task. NestForge can
contribute the two stencil files with its fixtures.

Two things the plain references use have no DaCe counterpart today: DaCe has no `Sendrecv` node, and its `mpi4py`
replacements cover neither `comm.Sendrecv` nor `comm.Shift`. The DaCe form therefore takes the two neighbor ranks as
scalar arguments (the bench's driver calls `MPI_Cart_shift` and passes them, or the entry computes them in a
tasklet), and writes each exchange as `Isend`, `Irecv`, and `Waitall`.

## B4: tuner episodes on a distributed task

**Missing.** `Task` carries `residency`, and `Residency.DISTRIBUTED` exists for agent submissions. A tuner never
receives it: `harness/run_tuner.py:task_of` builds `Task(kernel, source_mode, language, image=image)` and leaves the
residency at its default, and `FlavorOptimizer.library_submission` builds a single-node library.

**Proposed design.**

1. `task_of` takes the residency from the setup (`mpi.grade_distributed`), the way it takes `image` since #179.
2. A flavor declares `residencies = ["host", "distributed"]` in its manifest entry (default `["host"]`, plus `device`
   for a GPU column). The bench declines a task outside the list as unsupported (B8), so no existing flavor changes.
3. `Submission` needs no change. A distributed library exports `<base>_mpi`, and the judge picks the symbol by the
   task's residency.
4. The user-plugin CI gains one distributed stub episode on `jacobi_2d` at 2 ranks.

An ICON task with MPI ranks is a later request. It needs B1 to B4 first, and its own manifest.

## B5: layout requests on a distributed array

**Question.** A submission may carry layout operations (`permute`, `block`) and distribution operations
(`distribute`, `replicate`) on the same array. Which dimension names does `distribute.dim` refer to after a
`permute`: the original ones or the permuted ones? NestForge refuses `relayout` on a distributed argument until the
requests documentation states the order, with a test.

**Proposed answer.** Operations apply in list order, and each names dimensions as they are after the operations
before it. This is how the digit operations already compose.

## B6: precision per argument in the entry

**Missing.** The entry declares every floating-point argument at the kernel's one precision. A `relayout` cast (an
argument stored in a narrower type) cannot be finalized, and NestForge refuses it (`stages/layout.py:141`).

**Proposed design.** The `cast` request already exists for the `mixed-precision` request set. `render_canonical`
takes the submission's `requests` and declares each cast argument at its requested type, and the harness passes that
argument converted, untimed, as it does for a layout. The judge compares outputs at the kernel's precision after
converting back. NestForge then passes its cast operations through `Final.requests`, which it fills for layouts
today.

## B7: a tool call's route request names the assistant message

**Missing.** `RouteRequest` carries `method`, `parts`, `query`, `body`, and `episode_id`. A route cannot say which
assistant message made the call, so a plugin record cannot be joined to the usage table per message.

**Proposed design.** `RouteRequest` gains `message_index: int | None`, the index the transcript and the usage table
use, sent by the agent-side tool wrapper in a header. `None` for a call a tuner makes. Existing plugins read fields
by name and are unaffected.

## B8: a tuner flavor can decline a kernel as unsupported

**Missing.** A hand-written column without the kernel's file is recorded as unsupported since #184. A
`FlavorOptimizer` has no way to say the same, so a flavor that cannot take a kernel raises, and the episode counts as
a crash.

**Proposed design.** `plugin_api` exports an exception `Unsupported(reason)`. `tune` raises it, and the harness
records the episode as unsupported with the reason, in the same outcome class as #184.

## B9: a route request names the episode's packet

**Status.** Not needed for NestForge's design: the agent-side tool wrapper forwards the packet's setting with each
call ([move-families.md](move-families.md)). The request remains as hardening.

**Missing.** A plugin's session lives in the judge's process. A packet's `env` reaches the agent's tools, and
`RouteRequest` has no packet field, so a route learns a packet's setting only from what the tool wrapper sends.

**Proposed design.** `RouteRequest` gains `packet: str`, the resolved packet name of the episode ("" for a tuner). A
plugin reads its own packet table for anything further.

## B10: mypy error in `config.py`

`pre-commit run --all-files` on main reports an argument-type error at `hpcagent_bench/config.py:1094` (the
`TypeAdapter(Annotated[(...)])` call in `adapter`). Bench PR #190 fixes it.

## B11: a NestForge submission adds no communication

**Missing.** The paper claims that the agent never writes communication code. With the packet `nestforge-full` the
agent writes kernel code, and nothing stops a kernel from calling MPI.

**Proposed design.** Two checks, one on each side.

1. NestForge refuses a `write_kernel` source that references an MPI, NCCL, or RCCL symbol or header (a token check on
   the preprocessed unit, then the undefined symbols of the built object: no `MPI_*`, `PMPI_*`, `ncclXxx`).
2. The bench's anti-cheat gets a rule a packet can switch on, `communication = "canonical"`: the undefined MPI symbols
   of the submitted library must be a subset of those of the canonical program's library for the same kernel, which
   the judge builds once per kernel. A submission that adds a call fails the rule and is graded as a violation.

## B12: two application tasks, the ICON dynamical core and OMEN

**Goal.** The application experiment runs on two programs (owner's decision, 2026-11-16): the ICON dynamical core
and OMEN, the quantum-transport code whose kernel the layout study uses. Several agents optimize each program
independently, as on every other task, so each program is one bench task with a manifest, a reference, and a
distributed grade.

**What each task needs.** The same six items for both programs.

| Item | ICON dynamical core | OMEN |
|---|---|---|
| Program text | Fortran through dace-fortran into one SDFG | to be named by the owner: the DaCe Python form of the SC19 work, or the original source |
| Communication | halo exchanges as communication nodes | to be read from the program |
| Reference | the unmodified program's output on the task's input | the same |
| Input | one grid and one time step count per size preset | one device structure per size preset |
| Distribution | unstructured: see below | to be read from the program |
| Rank counts | from the setup's `mpi.rank_counts`, capped by the node count | the same |

**The distribution gap.** `harness/mpi_descriptor.py` describes block, block-cyclic, and cyclic ownership of array
axes over a Cartesian grid. ICON's grid is unstructured: each rank owns an index set from a graph partition, with a
halo list per neighbor. The bench cannot scatter or gather such arrays today. Three ways to close the gap:

1. A new scheme `indexed`. The manifest ships, per rank count, the owned index list of each array axis and the
   neighbor lists, produced offline by the application's own partitioner. The driver scatters and gathers by index
   list, and the kernel receives its halo lists as arguments.
2. The task runs the application's own setup and hands the kernel a communicator and the rank's patch. The bench then
   checks the output through the application's own gather.
3. One rank's patch on one node with the exchange stubbed. No bench change, and no scaling numbers.

Recommended: option 1 as the feature, with option 3 as the first milestone that exercises the whole pipeline.

**Open for the owner.** The source form of OMEN, the inputs and sizes of both tasks, and the option above.

## B13: an LLVM-pass agent packet

**Missing.** The NestForge study compares the agent working through NestForge's moves with agents that write code
(the default agent and AutoKernel). A reviewer will ask whether any catalogue of verified transformations does as
well, and Klepl et al. (arXiv 2608.08085) found that abstractions helped less than prompting. The bench has no
packet in which the agent may only choose compiler transformations. User decision, 2026-11-16: the study adds such a
baseline, and its agent chooses LLVM passes, their parameters, and `clang` flags.

**Proposed design.** A packet `llvm-passes` (CPU), modeled on AutoPass (arXiv 2606.20373):

1. The agent cannot edit source. As for the `nestforge` packet, the judge takes only a library
   (`JUDGE_INPUT_MODE=library`), and the native tools are `Read` only.
2. Tools, served judge-side:
   - `list_passes`: the new pass manager's pass names (`opt -passes=help` of the image's LLVM), their parameters, and
     the allowed flags;
   - `apply_pipeline(passes: str, flags: list[str])`: builds the task's starting program (below) as `clang -O0
     -Xclang -disable-O0-optnone -emit-llvm`, then `opt -passes=<passes>`, then `clang <flags>` to the library. It returns the
     time and the verdict of the bench's timed run, the optimization remarks (`-Rpass`, `-Rpass-missed`, and
     `-Rpass-analysis`, summarized per loop), and the build error if there is one;
   - `score` and `submit`, as for every packet. `submit` submits the library of the best correct pipeline, or the
     pipeline that the agent names.
3. Flags come from a list in the packet: `-march`, `-mtune`, the `-ffast-math` family, `-fopenmp`, and `-mllvm`
   options for the unroller, the vectorizer, and Polly (`-polly`, `-polly-parallel`). Polly counts as a pass, so the
   judge image's LLVM must be built with Polly. Any other flag is refused with the list.
4. The starting program is the task's canonical form without wavefronts, rendered as one C++ translation unit
   (DaCe's CPF rendering, `dace.codegen.cpf.cpf`), the same program the NestForge agent starts from
   ([skew-wavefront.md](skew-wavefront.md)). It needs DaCe's D3 switch, and the bench renders it as it renders the
   `cpf-cpu-src` packet's drop-in source today (`CPF_DROPIN_DIR`).
5. Legality: every pass preserves semantics. A flag that changes floating-point semantics is judged by the bench's
   oracle tolerance, as for every submission.
6. Cost and limits are those of every packet: the same turn cap, budget, and `usage.jsonl` accounting.

**Tests.** A stub agent runs `list_passes`, one pipeline (`-passes=default<O3>`), and `submit` on `jacobi_1d`, and
is graded correct. A refused flag returns the list. A pipeline that fails to build returns the error and leaves the
best library unchanged.

**Decided by the owner, 2026-11-16.** Polly counts as a pass. The starting program is the canonical form without
wavefronts, the NestForge agent's own starting point, so the two agents differ only in their catalogue.

The rendering keeps the OpenMP pragmas of the canonical form's doalls (owner, 2026-11-16).

## B14: a GEAK harness plugin

**Missing.** The study's GPU experiment has no baseline from the literature. GEAK (AMD, arXiv 2507.23194, open source,
v4 in 2026) optimizes HIP and Triton kernels on AMD Instinct GPUs, the study's hardware.

**First step, before any design.** Check whether GEAK can take a bench task: its input is a kernel repository, and a
bench task is a NumPy specification with a C ABI. If it can, the plugin is a harness like `plugins/optimas`, and it
runs one episode against the same judge under the same limits. The owner decides after that check (2026-11-16).
