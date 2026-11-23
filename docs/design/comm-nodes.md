# Communication nodes in NestForge

[Designs](README.md) · [Structure tree](../ir.md) · [Stages](../stages.md)

Status: motion rules built (2026-11-19, `tactics/ir/comm.py`, `stages/lib_motion.py`, `tests/test_comm_motion.py`):
the classes of section 1 and rules R1 to R5 of section 3, with `lib-into` refused. The tree rows of section 2 are built (2026-11-23, `comm_line` in `tactics/ir/comm.py`, `tests/test_comm_tree.py`). The invariants of section 4 are built (2026-11-23, `Session.comm_refusal`, `tests/test_comm_invariants.py`); `isolate_communication` is not. Section 5 is built (2026-11-23, `tests/test_comm_guards.py`): no row is a doall or joins a kernel, the default scopes cut at rows, `set_scope` refuses them, a kernel source may not call MPI, NCCL or RCCL, a program that communicates is placed on the CPU, and a row is not inlined and runs on MPI only. Section 6 is built (2026-11-23, `tests/test_comm_layout.py`): `relayout` refuses and does not list an argument the bench splits over the ranks (`Session.distributed`, from the spec's `mpi.arrays`); the buffers of communication rows, blocking and nonblocking, are relayout's to pack (`pack_mpi_buffers`, package 17), so they need no refusal. The rest is design, not implemented. Written 2026-11-16 against NestForge `bf068ae`, DaCe extended `561defb2a6`, and
HPCAgent-Bench `cc1124254`, and reviewed the same day against the code each section names.

## Goal

NestForge optimizes a program that communicates between MPI ranks. The agent moves the compute around the
communication and never writes or edits communication code. Every rank runs the same program, and a move is legal
only if the program stays correct at every rank count.

The target programs are two applications, the ICON dynamical core and OMEN (owner's decision, 2026-11-16). Several
agents optimize each program independently, as on every other task. The first milestone is smaller: `jacobi_2d` and
`heat_3d` with their halo exchanges.

Out of scope:

1. communication optimizations: no move rewrites a communication operation (a blocking exchange stays blocking, no
   neighborhood collectives, no RMA, no change of a datatype or a tag). Owner's decision, 2026-11-16;
2. GPU targets for a program that communicates (NCCL, RCCL, GPU-aware MPI);
3. a choice of the distribution by the agent. The bench's manifest sets the distribution.

The design has two parts. Sections 1 to 11 are the first version: communication rows are visible, they stay correct
under every move, and a program with them is timed and finalized. Section 12 is the overlap of communication with
compute by moving rows and statements. It is designed in detail here and built after the first version.

## What exists today

DaCe represents communication as library nodes of `dace.libraries.comm`. There are 26 node classes under one base
class, `CommNode`, and every one has an `MPI` implementation. A nonblocking operation writes a request, and a `Wait`
or `Waitall` completes it. `CommNode.has_side_effects` is true for the nodes that DaCe must neither remove nor
reorder (`Send`, `Recv`, `Isend`, `Irecv`, `Wait`, `Waitall`, `Barrier`, `Abort`, `CommSplit`, `CommFree`, and a
collective with `nonblocking` set). A blocking collective reports no side effects.

NestForge already treats a communication node as a library call, and four behaviors follow from that. Three of them
are wrong or incomplete for communication.

| Where | Today | Problem |
|---|---|---|
| `names.library_family` | the node is named `lib_comm<n>` | none: correct, and kept |
| `lib_motion.effect_refusal` | a node with side effects stays, and a blocking collective moves like a BLAS call | a collective can leave or enter a loop, and its order against other operations is not checked |
| `names.top_level_doalls` | every library call is a black-box doall, so it is a kernel of its own in the scopes phase | the agent could take a communication node into a kernel and rewrite it in the kernel phase |
| `stages.performance`, `stages.finalize` | the program runs in one process and is rendered in the single-node ABI | a program with communication nodes needs a launcher, a communicator, and the distributed ABI |

## Design

### 1. Classes of communication rows

A new module `tactics/ir/comm.py` classifies a node. Every rule below reads the class, never the node type.

| Class | Nodes | Meaning |
|---|---|---|
| `QUERY` | `CommRank`, `CommSize`, `CommF2c` | reads the communicator, sends nothing |
| `START` | `Isend`, `Irecv`, any collective with `nonblocking` set | starts an operation and writes a request |
| `WAIT` | `Wait`, `Waitall` | completes the operations whose requests it reads |
| `BLOCKING` | `Send`, `Recv`, and every collective without `nonblocking` (`Allreduce`, `Reduce`, `Bcast`, `Gather`, `Gatherv`, `BlockGather`, `Scatter`, `BlockScatter`, `Allgather`, `Alltoall`, `Alltoallv`, `ReduceScatter`, `Redistribute`) | starts and completes in one call |
| `FIXED` | `Barrier`, `Abort`, `CommSplit`, `CommFree`, `Dummy` | changes the communicator, synchronizes every rank, or holds state of the MPI runtime |

```python
class CommClass(enum.Enum): QUERY, START, WAIT, BLOCKING, FIXED

def comm_class(node: nodes.Node) -> CommClass | None        # None: not a communication node
def comm_rows(sdfg: SDFG) -> list[tuple[SDFGState, CommNode]]                     # program order, QUERY left out
def starts_of(sdfg: SDFG, wait: CommNode) -> list[tuple[SDFGState, CommNode]]
def buffers(state: SDFGState, node: CommNode) -> tuple[frozenset[str], frozenset[str]]   # (sent, received)
def skeleton(sdfg: SDFG) -> Skeleton                                              # section 4
```

`starts_of` returns the `START` rows a `WAIT` completes. A `START` writes its request through its `_request`
connector, and a `WAIT` reads requests through its `_request` connector. The `START` rows of a `WAIT` are those that
write the container the `WAIT` reads and that reach the `WAIT` in the control flow with no other `WAIT` on that
container between. When the request container is an array and the subsets cannot be compared, every `START` that
writes the array and reaches the `WAIT` counts. The answer may then hold too many rows, which keeps more buffers in
flight and is safe.

DaCe's `dace.libraries.comm.schedule` holds the same logic for schedule-tree nodes (`is_start`, `is_wait`,
`_wait_for_buffers_in_flight`). Those functions take tree nodes, so `comm.py` implements the SDFG form, and a test
compares the two on every fixture.

**One state per row.** DaCe keeps the order of two nodes in one state only when data flows between them, and a `WAIT`
has no edge to the buffers it completes. Canonicalization (`stages/canonicalize.py`) therefore gains a step
`isolate_communication`: every row other than `QUERY` gets a state of its own, in the order the program had. When the
order of two such rows inside one state is not fixed by the data flow, the program is refused with both labels. Every
later rule relies on this form.

### 2. Rows in the structure tree

A communication node keeps its `lib_comm<n>` label, so `Namer` and the move tables need no new kind. The row prints
`COMM` in place of `LIBNODE`, and adds the fields an agent needs to reason about an exchange.

```text
|- for0  t=0:TSTEPS
|  |- state3
|  |  `- lib_comm0  COMM Isend [MPI]  reads=['A'] request=req  to=down tag=0
|  |- state4
|  |  `- lib_comm1  COMM Irecv [MPI]  writes=['ghost_top'] request=req  from=up tag=0
|  |- state5  doalls=1  reads=['A'] writes=['B']
|  |- state6
|  |  `- lib_comm2  COMM Waitall [MPI]  completes=[lib_comm0, lib_comm1]
|  `- state7  doalls=1  reads=['A', 'ghost_top'] writes=['B']
```

| Field | On | Means |
|---|---|---|
| `request=<container>` | `START` | the request the operation writes |
| `completes=[labels]` | `WAIT` | the `START` rows it completes; the buffers of those rows are in flight until this row |
| `to=`, `from=`, `tag=` | point-to-point rows | the peer and the tag, printed from the memlets on the connectors |
| `op=` | reductions | the reduction operator |

`introspect.describe_node` gains one branch for `comm_class(node) is not None`. A `QUERY` row prints `COMM` too and
has none of the fields. A collapsed state counts its communication rows in a new field `comm=<k>`, apart from
`doalls=<k>`.

### 3. Legality of motion

`lib-before` and `lib-after` accept a communication row under five rules. `lib-into` is refused for every row other
than `QUERY`, since it would join the row into a state with other nodes ("a communication row keeps a state of its
own"). `lib_motion.effect_refusal` gains a branch that calls `comm.motion_refusal`, and each rule returns its own
reason text. A `QUERY` row moves like any other library call.

| Rule | Statement | Reason given to the agent |
|---|---|---|
| R1 order | A `START` or `BLOCKING` row never crosses another `START` or `BLOCKING` row. | "lib_comm3 would overtake lib_comm1; operations that start communication keep their order." |
| R2 region | A communication row other than `QUERY` never leaves or enters a `for`, an `if`, or a doall. | "lib_comm0 runs once per iteration of for0; moving it out changes how often it communicates." |
| R3 in flight | A `WAIT` row reads what its `START` rows send and writes what they receive. | follows from the existing rule "no block it crosses touches what it reads and writes" |
| R4 fixed | A `FIXED` row never moves. | "lib_comm5 (Barrier) synchronizes every rank; it stays." |
| R5 view | A row wired to a view or an aliasing array stays, as for every library call. | existing text |

R3 needs one change only: `effect_of` returns, for a `WAIT`, the union of the buffers of its `START` rows. This is the
rule NestForge already uses for a host `Sync`, whose effect is the device work it waits for. A block that writes a
buffer in flight would have to cross the `START` (which reads the buffer) or the `WAIT` (which now reads it too), and
both crossings are refused.

R1 keeps the order of every operation that starts communication. It is needed in two cases and stricter than MPI in
a third.

1. **Blocking point-to-point, needed.** Every rank runs `Send(up)` then `Recv(down)`. With the two exchanged, every
   rank waits in `Recv` for a message no rank has sent, and the program deadlocks.
2. **Two operations with the same peer, tag, and communicator, needed.** MPI matches them in the order they start. With
   two `Isend` rows exchanged, the receiver gets the two buffers swapped.
3. **Collectives, and point-to-point rows with provably different peers or tags, not needed.** Every rank runs the same
   program, so two independent collectives change places on all ranks at once, and messages with different tags
   never match each other.

The rule costs little. The gain of an overlap comes from moving a `WAIT` later and compute earlier, which R1 never
restricts, and exchanging two starts changes no time. R1 therefore holds for every class (owner's decision,
2026-11-16), as in DaCe's own pass (`dace.libraries.comm.schedule`).

A `WAIT` row may cross `START` and `BLOCKING` rows, since completing an operation later is always legal while its
buffers are untouched. R3 guards that move.

### 4. Invariants, checked after every change

The rules of section 3 cover the moves that name a communication row. Every other move could still change
communication without naming it. A fusion can pull a statement from behind a `WAIT` to a place before it, and the
`WAIT` has no data edge that would stop DaCe. The design therefore does not rely on each move's own check. Four
invariants are checked after every change that is about to be kept, in `Session.atomic`, for a session whose program
communicates. A change that breaks one is undone, and the call answers `illegal` with the invariant's reason.

| Invariant | Statement | Reason given to the agent |
|---|---|---|
| I1 skeleton | The communication skeleton is the same before and after. | "<kind> would change how lib_comm0 runs: <difference>." |
| I2 in flight | No block between a `START` and its `WAIT` writes a sent buffer or touches a received buffer. | "<kind> would let state7 read ghost_top before lib_comm2 completes the receive into it." |
| I3 isolation | Every row other than `QUERY` is alone in its state and outside every doall. | "<kind> would put lib_comm1 into a doall." |
| I4 kernels | No kernel holds a communication row. | "<kind> would take lib_comm0 into kernel k2." |

**The skeleton** (`comm.skeleton`) is the control-flow tree of the program with everything removed that holds no
communication row other than `QUERY`. What remains is, per region, its kind (`for`, a branch of an `if`, the top),
its iteration range or branch condition, and two ordered lists: its rows other than `WAIT`, and its `WAIT` rows.
Sub-regions keep their place in the first list. Two skeletons are equal when the trees match, the rows match by
label, and the ranges and conditions match symbolically. The name of a loop variable is not compared, since a fusion
renames it.

A `WAIT` may cross other rows (section 3), which is why it has a list of its own: where a `WAIT` sits among the other
rows is free, and I2 guards it.

The skeleton decides every case with one comparison.

| Change | Skeleton | Outcome |
|---|---|---|
| a move inside a region without a communication row | same | applies |
| `for-fusion` of the exchange's loop with a compute loop of the same range | same | applies, if I2 holds |
| `for-fission` that leaves all communication rows in one of the two loops | same | applies, if I2 holds |
| `for-fission` that separates communication rows of one loop | two loops where one was | refused |
| `interchange-*`, `hoist-invariant-guard`, `sink-invariant-guard`, `tile-for`, `untile-for`, `unroll-for` on a loop that holds a row | nesting, range, or count differs | refused |
| a move that turns such a `for` into a doall, `flatten-doalls`, vectorization of it | kind differs | refused (also I3) |
| `hoist-invariant-code` of a row other than `QUERY` | the row's region differs | refused |
| two `START` or `BLOCKING` rows change places | order differs | refused (R1) |
| a `WAIT` changes places with another row | same | applies, if I2 holds |

**I2** is computed from `starts_of`: for every `WAIT`, the blocks on any path from one of its `START` rows to it. Each
such block's read and write sets are compared with the buffers in flight, by container name in the first version and
by subset from section 12.1 on.

The invariants are the safety net, and they are cheap: one walk over the control-flow tree per kept change. The moves
still check what they can before they rewrite, so that `list_moves` offers no move the invariants would undo: a move
kind whose region holds a communication row is listed only when the table above says it can apply.

Two moves are refused on a communication row by name. `inline-library` is refused, since code generation expands the
row at finalize. `set-library-implementation` lists `MPI` only.

### 5. Scopes, kernels, and placement

A communication row is never a kernel and never part of one. This holds for `QUERY` rows too, so no kernel source
ever holds an MPI call.

- `names.top_level_doalls` leaves out every node with a communication class.
- `set_scope` refuses a selection that holds such a row: "communication stays outside kernels; scope the rows before
  and after lib_comm2 apart."
- The default scopes cut at communication rows, so the compute between two exchanges is one kernel candidate.
- The kernel phase lists no communication row, so `read_kernel` never shows MPI code. `write_kernel` refuses a source
  that names an MPI, NCCL, or RCCL symbol. [Bench request B11](bench-requests.md) is the judge's side of the same
  rule.
- Placement refuses a GPU target for a session with communication rows, with the reason "a program that communicates
  runs on the CPU in this version".

### 6. Layouts of exchanged arrays

`relayout` on an array that a communication row sends or receives changes whether the buffer is contiguous. DaCe packs
a non-contiguous buffer of a blocking operation through a transient, and refuses a non-contiguous buffer of a
nonblocking operation (`validate_nonblocking`).

- On an array in the buffers of a `START` row, `relayout` is listed only when the memlet stays contiguous in element
  order (`dace.libraries.comm.nodes.node.in_place`). Otherwise the move is refused with DaCe's reason.
- On an array in the buffers of a `BLOCKING` row, `relayout` applies. DaCe packs a buffer that is no longer contiguous
  when it expands the row at code generation, so the tree does not change, and the cost of the packing shows in the
  program's time.
- On an argument of the program that the bench distributes, `relayout` is refused. The bench owns the distribution of
  each array dimension, and a permutation would have to permute the bench's `distribute` request as well
  ([bench request B5](bench-requests.md)). The distributed arguments are those the kernel's manifest splits
  (`mpi.split` and `mpi.decomposition`). The plugin reads them from the bench's spec into a new field
  `Setup.distributed_arguments` (`tactics/server.py`).

### 7. The session: ranks, timing, and the reference

A session knows whether its program communicates: `Session.distributed` is true when any node has a communication
class. Three things change for such a session.

**Communicator.** The bench hands every rank a Cartesian communicator as the `comm` argument of the entry. DaCe's
nodes communicate over the handle on their `_comm` connector, else over a DaCe process grid, else over the world
(`mpi_communicator` in `dace/libraries/comm/nodes/node.py`). At session start, NestForge checks that the `_comm`
connector of every communication row leads to the entry's `comm` argument through a `CommF2c` row. A row on the world
communicator or on a process grid refuses the session, with the row's label, since the bench's driver owns the grid.

**Timing and correctness.** `get_performance` runs the program through the bench's distributed driver and never
through NestForge's in-process call. The bench owns `main`, `MPI_Init`, the scatter, the gather, and the timed
bracket, and it already compares the gathered output with the single-node reference. NestForge needs the judge's
`/profile` route to take rank counts ([bench request B2](bench-requests.md)). Per rank count, the answer holds the
time of the slowest rank, the verdict with the mismatched outputs, and the slowest rank's time inside MPI calls.

`ProgramReport` gains `ranks: int` and `comm_share: float | None` (the time inside MPI calls over the whole time).
`Performance` gains `scaling: list[RankPoint]`, filled when the tool is called with more than one rank count.

**Rank counts.** The `get_performance` tool takes `ranks: list[int] | None`. The default is one count: the largest of
the setup's `mpi.rank_counts` that is at most the number of cores of the episode's slot. Each rank gets
`cores // ranks` threads. A count outside the setup's list is refused with the list. Kernel reports (each kernel's own
time) come from the slowest rank of the run.

### 8. Finalize

`BenchTarget` gains `distributed: bool`. `finalize.assemble` asks the bench for the distributed form of the program,
the entry `<base>_mpi(local tiles, local scalars, MPI_Fint comm, workspace, workspace_size)` of the bench's ABI
contract, and builds it with the MPI flags the bench names ([bench request B1](bench-requests.md)). The `Submission`
is the library, as today. NestForge adds no `distribute` request, so the manifest's default distribution applies.

### 9. Inputs

Two frontends produce programs with communication nodes.

1. **Python.** DaCe's `mpi4py` replacements (`dace/frontend/python/replacements/mpi.py`) turn method calls on a
   communicator argument into communication nodes: `Send`, `Recv`, `Isend`, `Irecv`, `Bcast`, `Allreduce`, and
   `Alltoall` on an `Intracomm`, and `MPI.Request.Waitall`. They cover neither `comm.Sendrecv` nor `comm.Shift`, and
   DaCe has no `Sendrecv` node. The bench's distributed references are plain `mpi4py` and use both. Each therefore
   needs a DaCe form that writes its exchange as `Isend`, `Irecv`, and `Waitall` and takes the neighbor ranks as
   arguments ([bench request B3](bench-requests.md)).
2. **Fortran.** dace-fortran turns MPI calls into the same nodes (its tests `mpi_sendrecv_e2e_test.py`,
   `mpi_waitall_test.py`, `mpi_collectives_test.py`).

Canonicalization must keep the skeleton of section 4. This is a property of DaCe's canonicalization passes. Several
rely on `has_side_effects`, and a blocking collective reports none. One gap is known: `LoopToMap`
(`dace/transformation/interstate/loop_to_map.py`) has no check for side effects, so a loop around a blocking
`Allreduce` may become a map. Nothing tests it today.

The first work package settles this before anything else is built: it canonicalizes each fixture and compares the
skeleton before and after. A pass that changes the skeleton is fixed in DaCe on extended, with a regression test
there, and never worked around in NestForge.

**Settled on 2026-11-16 (DaCe extended `561defb2a6`): canonicalization breaks the skeleton.** Two probes show it.

- A loop of blocking `Allreduce` calls, one per row of an array, becomes a doall around the `Allreduce`.
- A loop holding a blocking `Send` and a blocking `Recv` is split into a doall of all sends, then a doall of all
  receives: the order changes, and blocking sends can deadlock.

A halo exchange (`Isend`, `Irecv`, `Waitall`) and a residual `Allreduce` in a time loop that carries a dependence
keep their skeleton. The cause has two parts: a blocking communication node reports no side effects
(`CommNode.side_effects` is false by design), and `LoopToMap` and the passes that split statements and distribute
loops do not consult side effects at all.

The owner's decision (2026-11-16): canonicalization is fixed on DaCe extended so that it does not move
communication nodes, or tasklets and library nodes with side effects, across one another or out of their loops,
and it still optimizes everything around them as far as it can. Every pass that changes gets a unit test. The fix
comes after `skew-wavefront`.

**Fixed on DaCe extended `6ed7dc19e6` (2026-11-16).** Every communication node reports a side effect, the queries of
a communicator (`CommRank`, `CommSize`, `CommF2c`) excepted. `loop_analysis.ordered_side_effects` finds such nodes in
a block, and `LoopToMap` (refusal `ORDERED_SIDE_EFFECT`), `LoopFusion`, `LoopFission`, the loop distribution, and
`WavefrontSkew` refuse to parallelize, interleave, separate, or skew them. Both probes keep their loop, and the
computation beside a collective still parallelizes (`tests/canonicalize/canonicalize_keeps_ordered_side_effects_test.py`).
NestForge's package 1 test follows once the bench pins this commit.

### 10. Agent-facing text

No new packet and no new phase. When the session is distributed:

- the moves prompt gains one paragraph: what a `COMM` row is, that it keeps its order, its loop, and its state, and
  that the compute around it is optimized as on any other program;
- the skill page gains a section "Programs that communicate" with the tree excerpt of section 2;
- `list_moves` on a communication row lists `lib-before` and `lib-after` only.

### 11. What the first version does not do

The first version makes a communicating program safe to optimize. It does not overlap an exchange with compute. A
`WAIT` moves behind a block only when that block touches none of the containers in flight, and a stencil keeps its
ghost cells and its interior in one array. Section 12 designs what the overlap needs.

### 12. Overlap of communication and compute (designed, built later)

An exchange overlaps with compute when independent work runs between its `START` rows and its `WAIT`. No operation
is rewritten: rows and statements change places, and the exchange stays the one the program had. The pieces below
are each useful without the others.

#### 12.1 Effects by subset

`lib_motion.crossed_refusal` compares container names. The halo fixture holds ghost rows and interior rows in one
array `Ap`, so by name the interior update conflicts with the receive into the ghost row.

- `Effect.reads` and `Effect.writes` become `dict[str, subsets.Subset]`: per container, the union of the subsets of
  the memlets (a bounding box when the union is not a range).
- A block's effect is computed the same way from the memlets on its access nodes, after propagation. For a loop or
  another region, a subset that names a symbol the region assigns falls back to the whole container.
- Two effects conflict on a container only when their subsets are not provably disjoint. The proof is DaCe's
  `dace.libraries.comm.nodes.node.disjoint(subset_a, subset_b, facts)`, under the symbol facts of the SDFG
  (`N >= 3`, loop bounds). Anything unproven is a conflict, so the answer by name stays the default.
- A refusal names the overlap: "state7 reads Ap[0, 1:N-1], which lib_comm1 receives into Ap[0, 0:N]".
- A view, an aliasing array, or a memlet without a subset falls back to the whole container.

The change is confined to crossings where one side is a communication row, and to invariant I2. Every other use of
`lib-*` keeps the check by name, so no existing move gets a new answer.

#### 12.2 Moving and splitting statements

One move on compute rows, a general move in the family `fission` ([move-families.md](../move-families.md)).

**`split-doall`** cuts a doall's range in two. Labels: one doall. Argument: `at = {"<param>": "<expr>"}`.

- The result is two doalls of the same body over `[lo, at)` and `[at, hi)`, side by side in the doall's state (user
  2026-11-22: no state split as a move; a communication row moved next to a doall splits the state, and a state
  holding a communication row is never fused, since its node has side effects). The parts
  are `[lo, Min(at - 1, hi)]` and `[Max(at, lo), hi]`; a `Min` or `Max` that the SDFG's facts prove redundant is
  left out, so a cut inside the range gives plain bounds, and an empty part is legal.
- It is always legal on a doall: the iterations are independent, so any partition of the range is the same
  computation. A doall with a `reduce=` field is split into two accumulations into the same target, which is legal for
  the same reason a doall may run in any order.
- A step other than 1 is refused in the first build, and so is a doall nested in another doall.
- The doall's label is retired, and the two parts get fresh labels.
- For a stencil over rows `1:rows+1`, two splits (`at = {"i": 2}`, then `at = {"i": "rows"}` on the second part) give
  the first row, the interior, and the last row. Only the first and the last row read ghost rows.
- `list_moves` on a doall next to an exchange offers the split points that separate what the buffers in flight
  touch: the bounds of the received subsets along each parameter, computed from 12.1. The agent can pass any other
  expression.

**`block-before` and `block-after`: dropped (user 2026-11-22).** With effects by subset (12.1), `lib-after` moves a
`WAIT` row past a state that touches no buffer in flight, which is the same overlap. The agent, or a reference
solution, writes it as a sequence: `split-blocking` on the exchange, `split-doall` on the stencil into boundary and
interior, and `lib-after` on each wait past the interior doall (its state splits there), which leaves the boundary after
the wait.

#### 12.3 Overlap is motion, not a move (user 2026-11-21)

An overlap is a `START` row moved earlier and its `WAIT` moved later. `lib-before` and `lib-after` already do this
under R1 to R5, so NestForge gets no `overlap-communication` move. Two moves make the motion reachable:

- **`split-blocking`** turns a `BLOCKING` row into a `START` row and a `WAIT` row in place: `Send` into `Isend` and
  `Wait`, `Recv` into `Irecv` and `Wait`, and a collective into its `NonblockingCommNode` form (`nonblocking=True`) and
  `Wait`, with a fresh request transient. Labels: the blocking row. Always legal: the two rows sit next to each other,
  so the program is the same until motion separates them. The two rows get fresh labels.
- **`split-doall`** (12.2) cuts a doall that touches both a buffer in flight and other data, so that the independent
  part can move between the start and the wait.

The refusal of a `lib-after` on a wait names the statement that stops it (12.1), which tells the agent what to split.
DaCe's `dace.libraries.comm.schedule.overlap_communication` stays the reference for the rules and the source of
further fixtures; NestForge does not call it (a schedule tree round trip renames rows).

#### 12.4 Time inside communication (owner 2026-11-23: B2's total, no per-row times)

The number an overlap reduces is the time the slowest rank spends inside MPI calls. Bench request B2 measures it:
`/profile` with `ranks` runs the program at each rank count as a grade runs it and reports, per count, `time_us`
(the timed bracket on the slowest rank) and `mpi_time_us` (time inside MPI calls on that rank, from a PMPI shim).
The bench's own agent-side `profile` does not offer `ranks`, so NestForge's `profile` adds it on a distributed task
and shows one row per rank count. `Performance` stays a single-node measurement: it runs in the judge's process
without MPI. Times per communication row (DaCe timer instrumentation per row) wait until an agent needs them.

#### 12.5 Who calls the overlap

The default pipeline never splits or moves communication rows (owner's decision, 2026-11-16). `run_defaults` leaves
communication rows where the program had them, so `nestforge_default` shows no overlap. A reference solution or an
agent calls the move and the splits.

#### 12.6 Tests

1. Effects by subset: the interior update and the ghost-row receive do not conflict, the boundary update and the
   receive do, and an unproven pair conflicts.
2. A crossing without a communication row gives the same answer as before the change, on every existing test.
3. `split-doall` at 2 and at `rows` yields three doalls whose ranges partition the original, and the program's output
   is bit-identical. A split of a doall with `reduce=` stays within the oracle's tolerance.
4. `split-doall` with `at` outside the range yields one empty part and the same output.
5. `lib-after` on a wait is refused across a state that reads what the wait completes, with the container and subset named.
6. `split-blocking` on every blocking row of the rank fixtures leaves the output bit-identical; on the halo
   fixture, `lib-after` on the wait is refused before the splits (reason names the subset) and legal after them.
7. After the overlap sequence, the four invariants of section 4 hold.
8. At 1, 2, and 4 ranks the overlapped program is correct, and one `rollback` restores the program's hash.
9. On a distributed task, `profile` offers `ranks`, sends them to the judge, and shows one row per rank count with
   `mpi_time_us`; a single-node task offers no `ranks`.

#### 12.7 Work packages

| # | Package | Files | Depends on |
|---|---|---|---|
| 11 | Effects by subset, for crossings with a communication row and for I2 | `stages/lib_motion.py`, `ir/comm.py`, `tests/test_comm_subsets.py` | 5 |
| 12 | `split-doall` | `stages/moves.py`, `docs/stages.md`, `tests/test_split_doall.py` | none |
| 13 | dropped (user 2026-11-22): `lib-after` on the wait does it, after 11 | | |
| 14 | `split-blocking` | `stages/lib_motion.py`, `ir/comm.py`, `tests/test_comm_overlap.py` | none |
| 15 | Time inside communication (12.4): `ranks` on NestForge's `profile` | `plugin/tactics/tools/nestforge.py`, `plugin/tactics/answers.py`, `tests/test_bench_plugin.py`, `tests/test_answer_text.py` | 8, bench B2 |
| 16 | Reference solutions with the overlap for `jacobi_2d` and `heat_3d` | `reference_solutions/` | 10, 11, 12, 14 |
| 17 | Relayout around an exchange (12.8): test on `redistribute`, fix refusals | `tests/test_comm_ranks.py` | none |
| 18 | Slice views of communication buffers removed in canonicalization (`RemoveSliceView` on a view that only feeds a `CommNode`; user 2026-11-22), so R5 no longer pins `Isend`/`Irecv`/`Allreduce` on a slice; a view that cannot become a pointer (strided) stays and stays pinned | `stages/canonicalize.py`, `tests/test_comm_motion.py`, `tests/comm_rank_programs.py` | 5 |

Package 12 has no dependency on communication and can be built at any time.

#### 12.8 Layouts around an exchange (OMEN, SC19)

OMEN computes in one layout, transposes locally, exchanges in another layout, and transposes back. NestForge adds no
move for it: the `relayout` and `relayout_local` moves reuse relayout's passes. `relayout.passes.sdfg.pack_mpi_buffers`
routes an MPI buffer of a laid-out array through a packed transient (pack before a send, unpack after a receive) and
keeps it in place where both ends share a layout; `library_rules` leaves `CommNode` to that pass. The pack and unpack
then appear as ordinary rows that motion and fusion treat like any other. The `redistribute` fixture of
`tests/comm_rank_programs.py` is the test; a refusal there is a bug to fix in the existing moves.

## Work packages of the first version

Each package is one pull request with its tests. The order is the dependency order.

| # | Package | Files | Depends on |
|---|---|---|---|
| 1 | Fixtures and the canonicalization test: a `jacobi_2d` halo (`Isend`, `Irecv`, `Waitall`), a residual `Allreduce` in the time loop, a ring of `Send` and `Recv` | `tests/comm_programs.py`, `tests/test_comm_canonical.py` | none |
| 2 | `ir/comm.py` (classes, `comm_rows`, `starts_of`, `buffers`, `skeleton`) and `isolate_communication` | `tactics/ir/comm.py`, `stages/canonicalize.py`, `tests/test_comm_classes.py` | 1 |
| 3 | Tree rows and fields | `ir/introspect.py`, `docs/ir.md`, `tests/test_comm_tree.py` | 2 |
| 4 | The four invariants after every kept change | `session.py`, `ir/comm.py`, `tests/test_comm_invariants.py` | 2 |
| 5 | Motion rules R1 to R5, and `lib-into` refused | `stages/lib_motion.py`, `tests/test_comm_motion.py` | 2 |
| 6 | Scopes, kernels, placement, and what `list_moves` offers | `stages/moves.py`, `stages/scopes.py`, `ir/names.py`, `stages/placement.py`, `stages/kernel.py`, `tests/test_comm_guards.py` | 4 |
| 7 | Layout rule | `stages/layout.py`, `server.py`, `tests/test_comm_layout.py` | 2 |
| 8 | Session: communicator check, timing at rank counts, reports | `session.py`, `stages/performance.py`, `tests/test_comm_performance.py` | bench B2 |
| 9 | Finalize in the distributed ABI | `stages/finalize.py`, `tests/test_comm_finalize.py` | bench B1 |
| 10 | Plugin: prompt paragraph, skill section, tool argument `ranks`, reference solutions for the two stencils | `plugin/tactics/`, `reference_solutions/` | 8, 9, bench B3, B4 |

Packages 1 to 7 need nothing from the bench and can start now. Package 1 comes first: if canonicalization changes the
skeleton, the DaCe fix is on the critical path of everything else.

## Tests of the first version

Tests assert structure before values, and each refusal test asserts the reason text.

1. Canonicalization keeps the skeleton of every fixture, and every row other than `QUERY` ends alone in its state.
2. Each of the 26 node classes has a class, and a new DaCe node without one fails the test by name.
3. `starts_of` agrees with DaCe's schedule-tree logic on every fixture.
4. The tree prints `request=`, `completes=`, `to=`, `from=`, and `tag=` for the halo fixture.
5. One refusal test per rule R1 to R5, and one for `lib-into`.
6. A `WAIT` moves behind a state that touches no container in flight, and is refused behind one that reads a received
   container.
7. One test per invariant with a change that breaks it: the change is undone, the program's hash is the one before,
   and the reason names the invariant's rows. For I2 the change is a fusion that pulls a reader of a received buffer
   to a place before the `WAIT`.
8. Every row of the table in section 4 has a test with the outcome the table states.
9. `set_scope` refuses a communication row, the default scopes cut at it, and `write_kernel` refuses a source that
   calls `MPI_Send`.
10. `relayout` is refused on a non-contiguous nonblocking buffer and on a distributed argument.
11. A session on the world communicator is refused when it starts.
12. At 1, 2, and 4 ranks, the halo fixture stays correct after a random sequence of listed moves (marker `mpi`, small
    sizes, the launcher of the bench's configuration).
13. The finalized library of the halo fixture is graded correct by an in-process judge at 2 ranks (marker `judge`).

A new marker `mpi` joins `pyproject.toml`: "launches MPI ranks; its own CI job".

## Done means

1. The thirteen tests pass in CI.
2. `nestforge_reference` on `jacobi_2d` and `heat_3d`, distributed, is graded correct at every rank count of the
   setup. Its time is reported next to the bench's C reference. The first version promises no speedup of the
   exchange.
3. A stub agent episode on `jacobi_2d` runs in the bench's user-plugin CI.
4. `docs/ir.md`, `docs/stages.md`, the README, and the skill page describe the communication rows.

## Decisions

Taken by the owner on 2026-11-16:

1. No communication optimizations. A move never rewrites a communication operation, so there is no `set-nonblocking`.
2. The overlap is done by moving communication rows and statements (section 12). It is designed here and built after
   the first version.
3. R1 is strict for every class (section 3).
4. The default pipeline never splits or moves communication rows (section 12.5).
5. The application targets are the ICON dynamical core and OMEN, each optimized by independent agents.

Open:

1. GPU targets for programs that communicate. Recommended: after the CPU results.
2. How ICON and OMEN become bench tasks ([bench request B12](bench-requests.md)).
