# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every NestForge tool: its phase, its arguments and what the model is told. Pure pydantic, so the agent image
(no DaCe, no NestForge) and the judge route read the same table."""

import enum
from typing import Any, Literal, NamedTuple

import pydantic

__all__ = ["IDENTITY", "ROUTE", "TOOLS", "Args", "Effect", "NoArgs", "Phase", "ToolDef"]

#: The judge route every tool posts to, as ``/nestforge/<tool>``.
ROUTE = "nestforge"

#: Body fields the transport adds to every request; never a tool argument.
IDENTITY = frozenset({"kernel", "rank", "episode_id", "optimizer", "tokens", "disabled_families"})


class Phase(enum.StrEnum):
    """A tool's phase: the id of its ``[phases.<id>]`` in plugin.toml, or any phase."""

    MOVES = "moves"
    SCOPES = "scopes"
    KERNELS = "kernels"
    CODEGEN = "codegen"
    ANY = "any"


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class NoArgs(Args):
    pass


class GetWorkDepth(Args):
    labels: list[str] = pydantic.Field(
        default_factory=list, description="Rows to give work and depth for (loops, doalls, states); none: every row."
    )
    formula: bool = pydantic.Field(
        False, description="Give the formulas in the sizes instead of their values at this session's sizes."
    )


class ViewTree(Args):
    expand: list[str] = pydantic.Field(
        default_factory=list, description="State or doall labels to open, with everything below them."
    )
    collapse: list[str] = pydantic.Field(
        default_factory=list,
        description="Open state or doall labels to close, after expand: one inside an expanded row closes alone.",
    )
    c: list[str] = pydantic.Field(
        default_factory=list,
        description="Row labels whose leaf bodies print as C instead of Python (rows inside them too, unless set "
        "themselves): C shows C++ tasklets (vector intrinsics, inline assembly) as written. Read-only.",
    )
    python: list[str] = pydantic.Field(
        default_factory=list, description="Row labels whose leaf bodies print as Python again (the default)."
    )


class ListMoves(Args):
    kind: str | None = pydantic.Field(None, description="Only moves of this kind.")


class Labeled(Args):
    labels: list[str] = pydantic.Field(description="Tree labels as view_tree printed them.")


class TuneMove(Labeled):
    kind: Literal["tile-doall", "tile-for", "vectorize-doall", "set-schedule-gpu"] = pydantic.Field(
        description="The move to try sizes of: tile sizes (tiling), lanes (vectorize-doall), block (set-schedule-gpu)."
    )
    values: list[list[int]] = pydantic.Field(
        min_length=1,
        max_length=4,
        description="1 to 4 values of the kind's sizes, each as apply_move takes them, e.g. [[64], [128], [256]].",
    )
    scope: str | None = pydantic.Field(
        None, description="The kernel, as apply_move takes it; set-schedule-gpu needs one."
    )


class ApplyMove(Labeled):
    kind: str = pydantic.Field(description="A move kind list_moves offered.")
    tile_sizes: list[int] = pydantic.Field(
        default_factory=list,
        description=(
            "tile-doall: one tile size per dimension, e.g. [32, 32]; tile-for: one size, e.g. [64]; skew-wavefront: "
            "two extents, along the loop and along the diagonal."
        ),
    )
    lanes: list[int] = pydantic.Field(
        default_factory=list,
        description="vectorize-doall: the lanes per vectorized dimension, innermost last, e.g. [8] (sve ignores them).",
    )
    block: list[int] = pydantic.Field(
        default_factory=list,
        description=(
            "stage-loads-shared, and set-schedule-gpu on a kernel doall: the thread block, one size per dimension "
            "(at most 3), e.g. [256] or [32, 8]."
        ),
    )
    implementation: str | None = pydantic.Field(
        None, description="set-library-implementation: one implementation list_moves offered for the row."
    )
    max_iterations: int = pydantic.Field(
        16, description="unroll-for, unroll-doall: the largest trip count you accept to unroll; list_moves checks 16."
    )
    config_values: dict[str, int | float] = pydantic.Field(
        default_factory=dict,
        description="propagate-config (labels []): the values of scalar arguments and free symbols you expect at run "
        'time, e.g. {"K": 3}.',
    )
    isa: str = pydantic.Field(
        "auto",
        description="vectorize-doall: the ISA of the tiles, one of auto, avx512, avx2, sve, neon, scalar.",
    )
    force: bool = pydantic.Field(
        False,
        description="for-to-doall: apply although the dependence check refused, because you know that no two "
        "iterations write one element (e.g. an index array is a permutation). DaCe's checks of conflicting writes and "
        "some of carried reads are waived; a break, continue or return in the body, a communication call, and a read "
        "DaCe ties to another iteration's write are still refused. The answer carries a warning, "
        "and a wrong program is yours to find.",
    )
    schedule: Literal["static", "dynamic", "guided", "default", "sequential"] | None = pydantic.Field(
        None,
        description="set-schedule-cpu (inside a CPU kernel): the OpenMP schedule of the doall (static, dynamic, "
        "guided, or default: DaCe chooses). set-schedule-gpu, on a doall directly inside a GPU kernel's doall: how the block's threads "
        "share it (sequential: each thread runs it whole; static: one iteration per thread; dynamic: a shared queue). "
        "dynamic or guided pays when iterations differ in cost (a triangular nest, sparse rows).",
    )
    chunk: int | None = pydantic.Field(
        None, description="set-schedule-cpu: the iterations a thread takes at a time, e.g. 16; none: OpenMP's default."
    )
    threads: int | None = pydantic.Field(
        None, description="set-schedule-cpu: the threads that run the doall; none: all of them."
    )
    container: str | None = pydantic.Field(
        None,
        description="sink-invariant-code: the temporary whose statements move into the loop or doall, as list_moves "
        "names it (the state's writes=[..] shows it).",
    )
    operation: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description="relayout (labels []): one operation of the bench's requests on an array argument, as relayout "
        'reads it, e.g. {"action": "permute", "array": "A", "order": ["i1", "i0"]} or {"action": "block", '
        '"array": "A", "digit": "i1", "factor": "8", "into": ["ib", "ii"]}; digits are named i0, i1, ... before any '
        "operation renames them.",
    )
    # list_moves' note on a listed move, so the move is sent back as listed; never passed to the session
    current: str | None = pydantic.Field(
        None, exclude=True, description="Ignored: the implementation or schedule list_moves said the row has now."
    )


class InScope(Args):
    scope: str = pydantic.Field(description="The kernel whose body holds the rows, as list_kernels names it.")


class ListScopeMoves(ListMoves, InScope):
    pass


class ApplyScopeMove(ApplyMove, InScope):
    pass


class RunDefaults(Args):
    stage: Literal["moves", "finish", "scopes", "placement", "round"] = pydantic.Field(
        description="moves: default fusion and the stages after it; finish: the stages after your own moves; "
        "scopes: the default kernels (parallel maps; sequential and mixed loops whole); placement: fixes the kernels "
        "(what no scope covers gets its default scope) and puts parallel kernels on the GPU when it is a target; "
        "round: back to the program as the moves left it, dropping scopes and kernel code."
    )


class UnsetScope(Args):
    name: str = pydantic.Field(description="The kernel to inline back, as list_kernels names it.")


class Place(Args):
    devices: dict[str, Literal["cpu", "gpu"]] = pydantic.Field(
        default_factory=dict, description="Kernel name -> cpu or gpu; kernels left out keep the default device."
    )


class KernelName(Args):
    name: str = pydantic.Field(description="A kernel list_kernels named.")


class WriteKernel(KernelName):
    source: str = pydantic.Field(
        description='The whole translation unit, with the entry read_kernel showed (same extern "C" name and '
        "parameters, in order); a GPU kernel's host and device units joined as read_kernel joined them."
    )
    language: Literal["cpp", "cuda", "hip"] = pydantic.Field(
        description="cpp for a CPU kernel; for a GPU kernel the dialect read_kernel showed (cuda or hip)."
    )


class TuneKernel(KernelName):
    compilers: list[str] | None = pydantic.Field(
        None, description="Toolchains to try (gcc, clang, nvcc, hipcc, ...); every one found for the device if omitted."
    )


class Effect(enum.StrEnum):
    """What a call does to the session, which the agent side lists as MCP annotations."""

    READ = "read"  # changes nothing
    ADD = "add"  # changes the program or a kernel; an inverse or another call undoes it
    REPLACE = "replace"  # overwrites earlier work (a kernel's source, a scope's kernel)


class ToolDef(NamedTuple):
    phase: Phase
    args: type[Args]
    description: str
    effect: Effect = Effect.ADD


TOOLS: dict[str, ToolDef] = {
    "get_cpf": ToolDef(
        Phase.MOVES,
        NoArgs,
        "Start here: the program in canonical parallel form, as a labeled tree.",
        Effect.READ,
    ),
    "view_tree": ToolDef(
        Phase.MOVES,
        ViewTree,
        "Open states and doalls with everything below them, close some again, print bodies as C or Python, then show "
        "the tree. Changes only what is shown, never the program.",
        Effect.READ,
    ),
    "get_work_depth": ToolDef(
        Phase.MOVES,
        GetWorkDepth,
        "Work (operations) and depth (the longest dependent chain) of tree rows, at this session's sizes or as "
        "formulas: rank loops by them before choosing what to move. The tree does not print them.",
        Effect.READ,
    ),
    "list_move_kinds": ToolDef(
        Phase.MOVES,
        NoArgs,
        "Every move kind: the tree rows it takes, the transformation it runs, what it does.",
        Effect.READ,
    ),
    "list_moves": ToolDef(
        Phase.MOVES,
        ListMoves,
        "Every legal move right now: fusion, fission, interchange, tiling, unrolling, library expansion and "
        "implementation, and propagate-config.",
        Effect.READ,
    ),
    "apply_move": ToolDef(
        Phase.MOVES,
        ApplyMove,
        "Apply one move as list_moves listed it (or with your own tile_sizes, lanes, block, max_iterations, config_values); answers "
        "at once with the outcome, not with a time: call get_performance after the moves you want to compare. It "
        "retires the labels of the rows it took: read the tree again before the next move. kind rollback, labels [], "
        "takes the last change back (the last 8 can be).",
    ),
    "tune_move": ToolDef(
        Phase.MOVES,
        TuneMove,
        "Try a tiling, vectorize-doall or set-schedule-gpu move at several sizes in one call, each measured as "
        "get_performance measures. The fastest "
        "size that beats the program now is kept (status kept); if none does, the program stays as it was. Use it "
        "instead of tiling, untiling and tiling again.",
    ),
    "run_defaults": ToolDef(
        Phase.MOVES,
        RunDefaults,
        "Finish one stage with NestForge's default heuristic; call get_performance to time it.",
    ),
    "set_scope": ToolDef(
        Phase.SCOPES,
        Labeled,
        "Make one kernel of several top-level doalls of one state, or of a straight run of blocks; answers with the "
        "kernel's name.",
    ),
    "unset_scope": ToolDef(Phase.SCOPES, UnsetScope, "Inline a kernel back into the program.", Effect.REPLACE),
    "list_scope_moves": ToolDef(
        Phase.SCOPES,
        ListScopeMoves,
        "list_moves inside one kernel's body; on a GPU kernel with its copies inserted also stage-loads-shared.",
        Effect.READ,
    ),
    "apply_scope_move": ToolDef(
        Phase.SCOPES,
        ApplyScopeMove,
        "apply_move inside one kernel's body, as list_scope_moves listed it. A placed kernel is placed again with the "
        "change; stage-loads-shared rewrites the GPU kernel as placed, so do it last: placing again drops it.",
    ),
    "place": ToolDef(
        Phase.SCOPES,
        Place,
        "Run each kernel on the CPU or the GPU; a GPU kernel copies its arguments to the device and back inside its "
        "call. Answers each kernel's device and the transfers between them.",
        Effect.REPLACE,
    ),
    "insert_copies": ToolDef(
        Phase.SCOPES,
        NoArgs,
        "Move the GPU kernels' host/device copies out of the calls into the program, so a value that stays on the "
        "GPU between kernels is copied once; answers the copies.",
    ),
    "list_kernels": ToolDef(Phase.KERNELS, NoArgs, "Every kernel with its device and arguments.", Effect.READ),
    "focus_kernel": ToolDef(
        Phase.KERNELS,
        KernelName,
        "Scope your kernel work to one kernel: write_kernel and tune_kernel take only the kernel in focus.",
    ),
    "read_kernel": ToolDef(Phase.KERNELS, KernelName, "A kernel's current source.", Effect.READ),
    "write_kernel": ToolDef(
        Phase.KERNELS,
        WriteKernel,
        "Replace a kernel's source; it is built, checked against the kernel's oracle and linked only when it matches "
        "(status unchecked: it has no oracle, so it is linked and the whole program checks it).",
        Effect.REPLACE,
    ),
    "tune_kernel": ToolDef(Phase.CODEGEN, TuneKernel, "Sweep compilers and flags for one kernel."),
    "get_performance": ToolDef(
        Phase.ANY,
        NoArgs,
        "Measure the program as it stands: whole-program time, offloading (devices, transfers) and per-kernel times, "
        "with hints. Kernels are built with one fixed configuration, so two measurements compare; finalize's compiler "
        "sweep makes the graded library faster than this. It waits for a device slot and takes minutes on the first "
        "call; kernels a move left unchanged are not measured again, and an unchanged program answers at once. A "
        "correct program more than 3% slower than the fastest one measured is rolled back to that one (the answer "
        "says how many changes), so no measured program is lost to a slower one; measure only when you want to keep "
        "the result.",
    ),
    "applied_moves": ToolDef(
        Phase.ANY, NoArgs, "Every change this program holds, in order: moves, defaults, scopes, kernels.", Effect.READ
    ),
    "finalize": ToolDef(Phase.ANY, NoArgs, "Build the program as one library in the bench C ABI."),
}
