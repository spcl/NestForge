# What NestForge needs from HPCAgent-Bench

[Designs](README.md)

Written 2026-11-16 for the bench maintainers, against HPCAgent-Bench `cc1124254` (on the way to v1.0) and NestForge
`bf068ae`. Each request says what is missing, why NestForge needs it, and a proposed design. NestForge patches none
of them on its side. B1 to B5 belong to the distributed work ([comm-nodes.md](comm-nodes.md)), and B6 to B10 are
independent of it.

| # | Request | Blocks | Size |
|---|---|---|---|
| B1 | The canonical form renders the distributed entry | distributed finalize | medium |
| B2 | `/profile` runs a library at given rank counts | distributed `get_performance` | medium |
| B3 | A DaCe form of each distributed reference | every distributed column | small per kernel |
| B4 | Tuner episodes on a distributed task | every distributed column | medium |
| B5 | Layout requests on a distributed array | `relayout` on distributed arguments | design question |
| B6 | Precision per argument in the entry | `relayout` casts | medium |
| B7 | A tool call's route request names the assistant message | token split per phase | small |
| B8 | A tuner flavor can decline a kernel as unsupported | `nestforge_reference` | small |
| B9 | A route request names the episode's packet | ablation packets | small |
| B10 | mypy error in `config.py` | clean hooks | small |

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

**Missing.** A plugin's session lives in the judge's process. A packet's `env` reaches the agent, and `RouteRequest`
has no packet field, so a route cannot read a setting of the episode's packet.

**Proposed design.** `RouteRequest` gains `packet: str`, the resolved packet name of the episode ("" for a tuner). A
plugin reads its own packet table for anything further.

## B10: mypy error in `config.py`

`pre-commit run --all-files` on main reports an argument-type error at `hpcagent_bench/config.py:1094` (the
`TypeAdapter(Annotated[(...)])` call in `adapter`). The cause is in main, and every pull request sees it. A typed
helper that builds the `Annotated` form, or a narrow `cast`, removes it.
