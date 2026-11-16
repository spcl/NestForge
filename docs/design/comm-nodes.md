# Communication nodes in NestForge

[Designs](README.md) · [Structure tree](../ir.md) · [Stages](../stages.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `bf068ae`, DaCe extended `561defb2a6`, and
HPCAgent-Bench `cc1124254`.

## Goal

NestForge optimizes a program that communicates between MPI ranks. The agent moves the compute around the
communication and never writes or edits communication code. Every rank runs the same program, and a move is legal
only if the program stays correct at every rank count.

The target experiment is a whole application (ICON through dace-fortran) on the bench's distributed tasks, measured
from 1 to 64 ranks. The first milestone is smaller: `jacobi_2d` and `heat_3d` with their halo exchanges.

Out of scope:

1. communication optimizations: no move rewrites a communication operation (a blocking exchange stays blocking, no
   neighborhood collectives, no RMA, no change of a datatype or a tag). Owner's decision, 2026-11-16;
2. GPU targets for a program that communicates (NCCL, RCCL, GPU-aware MPI);
3. a choice of the distribution by the agent. The bench's manifest sets the distribution.

The design has two parts. Sections 1 to 10 are the first version: communication rows are visible, they stay correct
under every move, and a program with them is timed and finalized. Section 11 is the overlap of communication with
compute by moving rows and statements. It is designed in detail here and built after the first version.

## What exists today

DaCe represents communication as library nodes of `dace.libraries.comm`. There are 26 node classes under one base
class, `CommNode`, and every one has an `MPI` implementation. A nonblocking operation writes a request, and a `Wait`
or `Waitall` completes it. `CommNode.has_side_effects` is true for the nodes that DaCe must neither remove nor
reorder (`Send`, `Recv`, `Isend`, `Irecv`, `Wait`, `Waitall`, `Barrier`, `Abort`, `CommSplit`, `CommFree`, and a
collective with `nonblocking` set). A blocking collective reports no side effects.

NestForge already treats a communication node as a library call, and four behaviors follow from that. Each one is
wrong or incomplete for communication.

| Where | Today | Problem |
|---|---|---|
| `names.library_family` | the node is named `lib_comm<n>` | correct, and kept |
| `lib_motion.effect_refusal` | a node with side effects stays, and a blocking collective moves like a BLAS call | a collective can leave or enter a loop, and its order against other operations is not checked |
| `names.top_level_doalls` | every library call is a black-box doall, so it is a kernel of its own in the scopes phase | the agent could take a communication node into a kernel and rewrite it in the kernel phase |
| `stages.performance`, `stages.finalize` | the program runs in one process and is rendered in the single-node ABI | a program with communication nodes needs a launcher, a communicator, and the distributed ABI |

## Design

### 1. Classes of communication rows

A new module `nestforge/ir/comm.py` classifies a node. Every rule below reads the class, never the node type.

| Class | Nodes | Meaning |
|---|---|---|
| `QUERY` | `CommRank`, `CommSize`, `CommF2c` | reads the communicator, sends nothing |
| `START` | `Isend`, `Irecv`, any collective with `nonblocking` set | starts an operation and writes a request |
| `WAIT` | `Wait`, `Waitall` | completes the operations whose requests it reads |
| `BLOCKING` | `Send`, `Recv`, and every collective without `nonblocking` (`Allreduce`, `Reduce`, `Bcast`, `Gather`, `Gatherv`, `BlockGather`, `Scatter`, `BlockScatter`, `Allgather`, `Alltoall`, `Alltoallv`, `ReduceScatter`, `Redistribute`) | starts and completes in one call |
| `FIXED` | `Barrier`, `Abort`, `CommSplit`, `CommFree`, `Dummy` | changes the communicator or synchronizes every rank |

```python
class CommClass(enum.Enum): QUERY, START, WAIT, BLOCKING, FIXED

def comm_class(node: nodes.Node) -> CommClass | None        # None: not a communication node
def starts_of(state: SDFGState, wait: CommNode) -> list[tuple[SDFGState, CommNode]]
def buffers(state: SDFGState, node: CommNode) -> tuple[frozenset[str], frozenset[str]]   # (sent, received)
def region_communicates(block: ControlFlowBlock) -> bool
```

`starts_of` follows the request container from a `WAIT` back to the `START` nodes that write it. DaCe's
`dace.libraries.comm.schedule` already holds this logic for schedule-tree nodes (`is_start`, `is_wait`,
`_wait_for_buffers_in_flight`). The implementer first checks whether DaCe can export the predicates on SDFG nodes. If
it can, NestForge calls them, and `comm.py` keeps only the class table.

### 2. Rows in the structure tree

A communication node keeps its `lib_comm<n>` label, so `Namer` and the move tables need no new kind. The row prints
`COMM` in place of `LIBNODE`, and adds the fields an agent needs to reason about an exchange.

```
|- for0  t=0:TSTEPS
|  |- lib_comm0  COMM Isend [MPI]  reads=['A'] request=req0  to=down tag=0
|  |- lib_comm1  COMM Irecv [MPI]  writes=['ghost_top'] request=req1  from=up tag=0
|  |- state4  doalls=1  reads=['A'] writes=['B']            <- interior, touches no ghost row
|  |- lib_comm2  COMM Waitall [MPI]  completes=[lib_comm0, lib_comm1]
|  `- state5  doalls=1  reads=['A', 'ghost_top'] writes=['B']
```

| Field | On | Means |
|---|---|---|
| `request=<container>` | `START` | the request the operation writes |
| `completes=[labels]` | `WAIT` | the `START` rows it completes; the buffers of those rows are in flight until this row |
| `to=`, `from=`, `tag=` | point-to-point rows | the peer and tag expressions, printed from the connectors' memlets |
| `op=` | reductions | the reduction operator |

`introspect.describe_node` gains one branch for `comm_class(node) is not None`. A collapsed state counts communication
rows in a new field `comm=<k>`, apart from `doalls=<k>`.

### 3. Legality of motion

`lib-before`, `lib-after`, and `lib-into` accept a communication row under five rules. `lib_motion.effect_refusal`
gains a branch that calls `comm.motion_refusal`, and each rule returns its own reason text.

| Rule | Statement | Reason given to the agent |
|---|---|---|
| R1 order | A `START` or `BLOCKING` row never crosses another `START` or `BLOCKING` row. | "lib_comm3 would overtake lib_comm1; operations that start communication keep their order." |
| R2 region | A communication row other than `QUERY` never leaves or enters a `for`, an `if`, or a doall. | "lib_comm0 runs once per iteration of for0; moving it out changes how often it communicates." |
| R3 in flight | A `WAIT` row reads what its `START` rows send and writes what they receive. | follows from the existing rule "no block it crosses touches what it reads and writes" |
| R4 fixed | A `FIXED` row never moves. | "lib_comm5 (Barrier) synchronizes every rank; it stays." |
| R5 view | Rows wired to a view or an aliasing array stay, as for every library call. | existing text |

R3 needs one change only: `effect_of` returns, for a `WAIT`, the union of the buffers of its `START` rows. This is the
rule NestForge already uses for a host `Sync`, whose effect is the device work it waits for. The rule is sufficient.
A block that writes a buffer in flight would have to cross the `START` (which reads the buffer) or the `WAIT` (which
now reads it too), and both crossings are refused.

R1 keeps the order of every operation that starts communication. It is needed in two cases and stricter than MPI
in a third.

1. **Blocking point-to-point, needed.** Every rank runs `Send(up)` then `Recv(down)`. With the two exchanged, every
   rank waits in `Recv` for a message no rank has sent, and the program deadlocks.
2. **Two operations with the same peer, tag, and communicator, needed.** MPI matches them in the order they start. With
   two `Isend` rows exchanged, the receiver gets the two buffers swapped.
3. **Collectives, and point-to-point rows with provably different peers or tags, not needed.** Every rank runs the same
   program, so two independent collectives change places on all ranks at once, and messages with different tags
   never match each other.

The rule costs little. The gain of an overlap comes from moving a `WAIT` later and compute earlier, which R1 never
restricts. Exchanging two starts changes no time. The first version therefore keeps R1 for every class, as DaCe's own
pass does (`dace.libraries.comm.schedule`). A relaxed R1 would have to prove case 3 from the peer and tag
expressions, which is a proof with no measured benefit behind it.

A `WAIT` row may cross `START` and `BLOCKING` rows, since completing an operation later is always legal while its
buffers are untouched. This is the move that lets compute overlap with communication, and R3 guards it.

### 4. Every other move

A move that changes how often or in which order a region runs must not change the communication inside it. The check
is one function, `comm.region_communicates(block)`, called from the precondition of the moves in the table.

| Moves | With a communication row inside the region |
|---|---|
| `for-fusion`, `for-fission`, `subgraph-fission` | allowed when the rows keep their relative order and stay in one loop of the same domain; fission never separates a `START` from its `WAIT` into loops of different iteration order |
| `interchange-*`, `tile-for`, `untile-for`, `unroll-for` | refused: the exchange would run in a different sequence or count |
| a move that turns a `for` into a doall, `collapse-doalls`, `tile-doall`, vectorization | refused: a doall is unordered |
| `hoist-invariant-for`, `hoist-invariant-doall` | a `QUERY` row hoists; every other class is refused (R2) |
| `expand-library` | refused on a communication row: code generation expands it at finalize |
| `set-library-implementation` | lists `MPI` only in the first version |
| `relayout` | section 6 |

The moves on a region without a communication row are unchanged. A halo exchange therefore splits a time loop into
the exchange and the compute, and the compute is optimized as before.

In the first version a `WAIT` moves behind compute only when that compute touches none of the containers in flight.
A stencil keeps its ghost cells and its interior in one padded array, so the first version cannot overlap its exchange.
Section 11 designs what that needs.

### 5. Scopes, kernels, and placement

A communication row is never a kernel and never part of one.

- `names.top_level_doalls` leaves out nodes with a communication class other than `QUERY`.
- `set_scope` refuses a selection that holds such a row: "communication stays outside kernels; scope the rows before
  and after lib_comm2 apart."
- The default scopes cut at communication rows, so the compute between two exchanges is one kernel candidate.
- The kernel phase lists no communication row, so `read_kernel` and `write_kernel` never show MPI code.
- Placement refuses a GPU target for a session with communication rows in the first version, with the reason "a
  program that communicates runs on the CPU in this version".

### 6. Layouts of exchanged arrays

`relayout` on an array that a communication row sends or receives changes whether the buffer is contiguous. DaCe packs
a non-contiguous buffer of a blocking operation through a transient, and refuses a non-contiguous buffer of a
nonblocking operation (`validate_nonblocking`).

- On an array in the buffers of a `START` row, `relayout` is listed only when the memlet stays contiguous in element
  order (`dace.libraries.comm.nodes.node.in_place`). Otherwise the move is refused with DaCe's reason.
- On an array in the buffers of a `BLOCKING` row, `relayout` applies, and the tree shows the packing copies DaCe adds
  as `lib_copy` rows.
- On an argument of the program that the bench distributes, `relayout` is refused in the first version. The bench
  owns the distribution of each array dimension, and a permutation would have to permute the bench's `distribute`
  request as well ([bench request B5](bench-requests.md)).

### 7. The session: ranks, timing, and the reference

A session knows whether its program communicates: `Session.distributed` is true when any node has a communication
class. Three things change for such a session.

**Communicator.** The bench hands every rank a Cartesian communicator as the `comm` argument of the entry. At session
start, NestForge checks that every communication row communicates over that argument (its `_comm` connector leads to
the entry's `comm` through a `CommF2c` row). A row on the world communicator or on a DaCe process grid is refused when
the session starts, with the row's label, since the bench's driver owns the grid.

**Timing and correctness.** `get_performance` runs the program through the bench's distributed driver and never
through NestForge's in-process call. The bench owns `main`, `MPI_Init`, the scatter, the gather, and the timed
bracket, and it already compares the gathered output with the single-node reference. NestForge needs the judge's
`/profile` route to take rank counts ([bench request B2](bench-requests.md)). Per rank count, the answer holds the
time of the slowest rank, the verdict with the mismatched outputs, and the slowest rank's time inside MPI calls.

`ProgramReport` gains `ranks: int` and `comm_share: float | None` (the time inside MPI calls over the whole time).
`Performance` gains `scaling: list[RankPoint]`, filled when the tool is called with more than one rank count.

**Time per communication row.** `comm_share` says how much of the step is communication, and the agent also needs to
know which row waits. `Performance` gains `comm_rows: list[CommRowReport]` with, per row, its label, its class, and
its time on the slowest rank. The time of a `WAIT` row is the time the program stood waiting, which is the number an
overlap reduces. The times come from DaCe's timer instrumentation on the communication rows of the trial copy, one
report per rank, reduced by the bench's driver with the maximum. If the bench's driver cannot collect per-rank
reports, the request is part of B2.

**Rank counts.** The `get_performance` tool takes `ranks: list[int] | None`. The default is the largest rank count of
the setup (`mpi.rank_counts`) that fits the node, so one call costs one run. A count outside the setup's list is
refused. Kernel reports (each kernel's own time) come from one rank of the run, the slowest.

### 8. Finalize

`BenchTarget` gains `distributed: bool`. `finalize.assemble` asks the bench for the distributed form of the program,
the entry `<base>_mpi(local tiles, local scalars, MPI_Fint comm, workspace, workspace_size)` of the bench's ABI
contract, and builds it with the MPI flags the bench names ([bench request B1](bench-requests.md)). The `Submission`
is the library, as today. NestForge adds no `distribute` request, so the manifest's default distribution applies.

### 9. Inputs

Two frontends produce programs with communication nodes.

1. **Python.** DaCe's `mpi4py` replacements (`dace/frontend/python/replacements/mpi.py`) turn `comm.Isend`,
   `comm.Allreduce`, and the like into communication nodes. The bench's distributed references are plain `mpi4py`
   today (`jacobi_2d_mpi.py`), so the corpus needs a DaCe form of each ([bench request B3](bench-requests.md)).
2. **Fortran.** dace-fortran turns MPI calls into the same nodes (its tests `mpi_sendrecv_e2e_test.py`,
   `mpi_waitall_test.py`, `mpi_collectives_test.py`).

Canonicalization must keep every communication row, its order against the other communication rows, and its loop.
This is a property of DaCe's canonicalization passes, several of which rely on `has_side_effects`, and a blocking
collective reports none. Two gaps in DaCe are known on 2026-11-16 and belong to the first work package.

- `LoopToMap` (`dace/transformation/interstate/loop_to_map.py`) has no check for side effects, and a blocking
  collective reports none. A loop around an `Allreduce` may therefore become a map. This is untested. The fixture
  with the residual `Allreduce` in the time loop settles it, and a wrong answer is a DaCe fix.
- DaCe has no `Sendrecv` node, and its `mpi4py` replacements cover neither `comm.Sendrecv` nor `comm.Shift`. The
  bench's `jacobi_2d_mpi.py` uses both. A DaCe form of that reference ([bench request B3](bench-requests.md)) either
  writes the exchange as `Isend`, `Irecv`, and `Waitall` and takes the neighbors as arguments, or waits for the two
  replacements in DaCe. Recommended: the first, since the nonblocking form is also the one the overlap needs. The first work package therefore includes a test that canonicalizes each fixture program and
compares the sequence of communication rows before and after. A pass that reorders or hoists a collective is a DaCe
fix on extended with a regression test, never a workaround in NestForge.

### 10. Agent-facing text

No new packet and no new phase. When the session is distributed:

- the moves prompt gains one paragraph: what a `COMM` row is, the five rules in one line each, and the overlap
  pattern of section 4;
- the skill page gains a section "Programs that communicate" with the tree excerpt of section 2;
- `list_moves` on a communication row lists `lib-before`, `lib-after`, and `lib-into` only.

### 11. Overlap of communication and compute (designed, built later)

An exchange overlaps with compute when independent work runs between its `START` rows and its `WAIT`. No operation
is rewritten: rows and statements change places, and the exchange stays the one the program had. Three pieces are
needed, and each is useful without the others.

#### 11.1 Effects by subset

`lib_motion.crossed_refusal` compares container names. The halo fixture holds ghost rows and interior rows in one
array `Ap`, so by name the interior update conflicts with the receive into the ghost row.

- `Effect.reads` and `Effect.writes` become `dict[str, subsets.Subset]`: per container, the union of the subsets of
  the memlets (a bounding box when the union is not a range).
- A block's effect is computed the same way from its states' read and write memlets after propagation
  (`SDFGState.read_and_write_sets` gives the names today; the subsets come from the memlets on the access nodes).
- Two effects conflict on a container only when their subsets are not provably disjoint. The proof is DaCe's
  `dace.libraries.comm.nodes.node.disjoint(subset_a, subset_b, facts)`, under the symbol facts of the SDFG
  (`N >= 3`, loop bounds). Anything unproven is a conflict, so the default stays the answer by name.
- A refusal names the overlap: "state5 reads Ap[0, 1:N-1], which lib_comm1 receives into Ap[0, 0:N]".
- A view, an aliasing array, or a memlet without a subset falls back to the whole container.

The change is confined to crossings where one side is a communication row. Every other use of `lib-*` keeps the
check by name, so no existing move gets a new answer.

#### 11.2 Moving and splitting statements

Two moves on compute rows. Both are general moves and belong to the `fission` and a new `motion` family
([move-families.md](move-families.md)).

**`split-doall`** cuts a doall's range in two. Labels: one doall. Argument: `at = {"<param>": "<expr>"}`.

- The result is two doalls of the same body over `[lo, at)` and `[at, hi)`, in two states in that order. `at` is
  clamped as `Max(lo, Min(at, hi))`, so an empty part is legal.
- It is always legal on a doall: the iterations are independent, so any partition of the range is the same
  computation. A doall with a `reduce=` field is split into two accumulations into the same target, which is legal for
  the same reason a doall may run in any order.
- A step other than 1 is refused in the first build.
- For a stencil over rows `1:rows+1`, two splits (`at = {"i": 2}`, then `at = {"i": "rows"}` on the second part) give
  the first row, the interior, and the last row. Only the first and the last row read ghost rows.
- `list_moves` on a doall next to an exchange offers the split points that separate what the buffers in flight
  touch: the bounds of the received subsets along each parameter, computed from 11.1. The agent can pass any other
  expression.

**`block-before` and `block-after`** move a state across its sibling blocks. Labels: the state and the anchor block.

- It is `lib-before` and `lib-after` for a whole state: the state's effect (11.1 when it crosses a communication row,
  by name otherwise) must not conflict with any block it crosses, the symbols it reads must not be assigned on an
  edge it crosses, and it stays in its region.
- A state that holds a communication row moves under that row's rules R1 to R5.
- A state never crosses a `FIXED` row.
- The move rewires the region's edges and keeps every condition and assignment on the edges it does not cross. A
  region whose blocks are not a chain (a branch in the edges) is refused.

With these two moves the agent, or the reference solution, writes the overlap as a sequence: split the stencil into
boundary and interior, move the interior between the starts and the wait, and leave the boundary after the wait.

#### 11.3 `overlap-communication`

One move that does the row motion of an exchange in one call. Labels: a `for` or the program's top region. It moves
rows only and never splits a statement.

1. In every chain of sibling blocks inside the labeled region, take the `START` rows in order. Move each one as far
   toward the start of the chain as `lib-before` allows, together with the pack copies that only it reads (the
   `lib_copy` rows that write its send buffer when that buffer is a transient no other block reads).
2. Take the `WAIT` rows in reverse order. Move each one as far toward the end of the chain as `lib-after` allows,
   together with the unpack copies that read only what it receives.
3. Stop at the region's boundary (R2), at a `FIXED` row (R4), and at another `START` or `BLOCKING` row for a start
   (R1).

The move is the fixed point of single `lib-before` and `lib-after` steps, so it adds no legality code: every step is
checked by the rules of section 3 with the effects of 11.1. The whole sequence is one atomic change and one journal
entry, and one `rollback` undoes it.

The answer lists what moved and what stopped each row:

```
applied: overlap-communication on for0
  lib_comm0 (Isend)   before state3     was after state3
  lib_comm2 (Waitall) after state4      was before state4; stopped by state5: reads Ap[0, 1:N-1], received by lib_comm1
```

The stopping reason tells the agent which statement to split (11.2) before calling the move again.

DaCe has a pass with the same goal on the schedule tree (`dace.libraries.comm.schedule.overlap_communication`).
NestForge does not call it: the conversion from a schedule tree back to an SDFG would rename and regroup rows, and the
tree's labels must survive a move. A test compares the order of communication rows NestForge reaches with the order
DaCe's pass reaches on each fixture.

#### 11.4 What the default pipeline does

`run_defaults` for the moves stage calls `overlap-communication` on the whole program after the default fusion. It
does not split statements. `nestforge_default` therefore shows the overlap that needs no insight, the reference
solutions show the overlap with the splits, and the agent is measured between the two.

#### 11.5 Tests

1. Effects by subset: the interior update and the ghost-row receive do not conflict, the boundary update and the
   receive do, and an unproven pair conflicts.
2. A crossing without a communication row gives the same answer as before the change, on every existing test.
3. `split-doall` at 2 and at `rows` yields three doalls whose ranges partition the original, and the program's output
   is bit-identical. A split of a doall with `reduce=` stays within the oracle's tolerance.
4. `split-doall` with `at` outside the range yields one empty part and the same output.
5. `block-before` is refused across a block that writes what the state reads, with the container and subset named.
6. `overlap-communication` on the halo fixture moves the waits behind the interior after the splits, and moves
   nothing before them, with the stopping reason asserted.
7. The order of communication rows equals the order DaCe's pass reaches.
8. At 1, 2, and 4 ranks the overlapped program is correct, and one `rollback` restores the program's hash.

#### 11.6 Work packages

| # | Package | Files | Depends on |
|---|---|---|---|
| 10 | Effects by subset, for crossings with a communication row | `stages/lib_motion.py`, `tests/test_comm_subsets.py` | 4 |
| 11 | `split-doall` | `stages/moves.py`, `docs/stages.md`, `tests/test_split_doall.py` | none |
| 12 | `block-before`, `block-after` | `stages/lib_motion.py` or a new `stages/block_motion.py`, `tests/test_block_motion.py` | 10 |
| 13 | `overlap-communication`, and its call in the default moves | `stages/moves.py`, `tests/test_comm_overlap.py` | 10 |
| 14 | Reference solutions with the overlap for `jacobi_2d` and `heat_3d` | `reference_solutions/` | 11, 12, 13, 9 |

Package 11 has no dependency on communication and can be built at any time.

## Work packages

Each package is one pull request with its tests. The order is the dependency order.

| # | Package | Files | Depends on |
|---|---|---|---|
| 1 | Fixtures and the canonicalization test: `jacobi_2d` halo (nonblocking), residual `Allreduce`, a ring of `Send` and `Recv` | `tests/comm_programs.py`, `tests/test_comm_canonical.py` | none |
| 2 | `ir/comm.py`: classes, `starts_of`, `buffers`, `region_communicates` | `nestforge/ir/comm.py`, `tests/test_comm_classes.py` | 1 |
| 3 | Tree rows and fields | `ir/introspect.py`, `docs/ir.md`, `tests/test_comm_tree.py` | 2 |
| 4 | Motion rules R1 to R5 | `stages/lib_motion.py`, `tests/test_comm_motion.py` | 2 |
| 5 | Guards on the other moves, scopes, kernels, and placement | `stages/moves.py`, `stages/scopes.py`, `ir/names.py`, `stages/placement.py`, `tests/test_comm_guards.py` | 2 |
| 6 | Layout rule | `stages/layout.py`, `tests/test_comm_layout.py` | 2 |
| 7 | Session: communicator check, timing at rank counts, reports | `session.py`, `stages/performance.py`, `tests/test_comm_performance.py` | bench B2 |
| 8 | Finalize in the distributed ABI | `stages/finalize.py`, `tests/test_comm_finalize.py` | bench B1 |
| 9 | Plugin: prompt paragraph, skill section, tool argument `ranks`, reference solutions for the two stencils | `plugin/nestforge/`, `reference_solutions/` | 7, 8, bench B3, B4 |

Packages 1 to 6 need nothing from the bench and can start now.

## Tests

Tests assert structure before values, and each refusal test asserts the reason text.

1. Canonicalization keeps the sequence of communication rows on every fixture.
2. Each of the 26 node classes has a class, and a new DaCe node without one fails the test by name.
3. The tree prints `request=`, `completes=`, `to=`, `from=`, and `tag=` for the halo fixture.
4. One refusal test per rule R1 to R5.
5. A `WAIT` moves behind a doall that touches no container in flight, and is refused behind one that reads a received
   container.
6. A block that writes a sent buffer is refused between the `START` and the `WAIT`, from either side.
7. Every move kind of the table in section 4 is refused, or applies, as the table says.
8. `set_scope` refuses a communication row, and the default scopes cut at it.
9. `relayout` is refused on a non-contiguous nonblocking buffer and on a distributed argument.
10. A session on the world communicator is refused when it starts.
11. At 1, 2, and 4 ranks, the halo fixture stays correct after a random sequence of listed moves (marker `mpi`, small
    sizes, the launcher of the bench's configuration).
12. The finalized library of the halo fixture is graded correct by an in-process judge at 2 ranks (marker `judge`).

A new marker `mpi` joins `pyproject.toml`: "launches MPI ranks; its own CI job".

## Done means

1. The twelve tests pass in CI.
2. `nestforge_reference` on `jacobi_2d` and `heat_3d`, distributed, is graded correct at every rank count of the
   setup and is at least as fast as the bench's `mpi4py` reference.
3. A stub agent episode on `jacobi_2d` runs in the bench's user-plugin CI.
4. `docs/ir.md`, `docs/stages.md`, the README, and the skill page describe the communication rows.

## Decisions

Taken by the owner on 2026-11-16:

1. No communication optimizations. A move never rewrites a communication operation, so there is no `set-nonblocking`.
2. The overlap is done by moving communication rows and statements (section 11). It is designed here and built after
   the first version.

3. R1 is strict for every class (section 3).

Open:

1. GPU targets for programs that communicate. Recommended: after the CPU results.
