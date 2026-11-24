# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every NestForge tool: its phase, its arguments and what the model is told. Pure pydantic, so the agent image
(no DaCe, no NestForge) and the judge route read the same table."""

import enum
from typing import Any, Literal, NamedTuple

import pydantic

__all__ = ["IDENTITY", "MOVE_DEFAULTS", "ROUTE", "TOOLS", "Args", "Effect", "NoArgs", "Phase", "ToolDef"]

#: The judge route every tool posts to, as ``/nestforge/<tool>``.
ROUTE = "nestforge"

#: Body fields the transport adds to every request; never a tool argument.
IDENTITY = frozenset({"kernel", "rank", "episode_id", "optimizer", "tokens", "disabled_families", "agent_phases"})
#: The packet setting naming the phases its agent runs, comma-separated; unset, every phase. Like the families, a
#: packet's ``env`` reaches only the agent's side, so every call carries it to the judge, which fixes it at the first.
AGENT_PHASES_ENV = "NESTFORGE_AGENT_PHASES"


class Phase(enum.StrEnum):
    """A tool's phase: the id of its ``[phases.<id>]`` in plugin.toml, or any phase."""

    STRUCTURED = "structured"
    SCOPES = "scopes"
    SCOPED = "scoped"
    KERNELS = "kernels"
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
    root: str | None = pydantic.Field(
        None,
        description="Show only this row and everything under it; none: the whole tree. A move's answer already "
        "carries the subtree it rewrote, so read the whole tree only when you lost track of it.",
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


class ExamplePair(Args):
    before: str = pydantic.Field(
        description="A small program before the transformation: a Python module with one @dace.program, or C code."
    )
    after: str = pydantic.Field(description="The same program after it: same parameters, same results.")
    language: Literal["python", "c"] = pydantic.Field(
        default="python", description="python (checked by running both on random inputs) or c (checked to compile)."
    )


class RequestTransformation(Args):
    name: str = pydantic.Field(description="A short kind name for the transformation, e.g. split-reduction-by-sign.")
    description: str = pydantic.Field(description="What it does and when it is legal, in a few sentences.")
    rows: list[str] = pydantic.Field(description="The rows of the current tree it would apply to.")
    examples: list[ExamplePair] = pydantic.Field(
        min_length=5, description="At least 5 pairs of small programs of your own, before and after; not the task."
    )


class SetLlvmPasses(Args):
    passes: str = pydantic.Field(description="An opt -passes= pipeline, e.g. default<O3> or function(loop-mssa(licm)).")


class SetMlirPasses(Args):
    pipeline: str = pydantic.Field(
        description="A polygeist-opt --pass-pipeline=, e.g. builtin.module(affine-loop-tile{tile-size=32},canonicalize)."
    )


class ApplyMoves(Args):
    moves: list["ApplyMove"] = pydantic.Field(
        min_length=1, description="The moves in order, each as apply_move takes it (labels as the tree has them then)."
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
    depth: int = pydantic.Field(
        2,
        description="set-pipeline-depth: the stages of the loop's pipeline, the buffers its tiles rotate through (2+).",
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
        "share it (sequential: each thread runs it whole; static: one iteration per thread; dynamic: a shared queue; "
        "guided: the shared queue, taking half of what is left of a range per round). "
        "dynamic or guided pays when iterations differ in cost (a triangular nest, sparse rows).",
    )
    chunk: int | None = pydantic.Field(
        None,
        description="set-schedule-cpu: the iterations a thread takes at a time, e.g. 16; none: OpenMP's default. "
        "set-schedule-gpu with dynamic or guided: the round widths (a block's or a warp's threads) a round takes at "
        "least, e.g. 2; none: a whole range.",
    )
    threads: int | None = pydantic.Field(
        None, description="set-schedule-cpu: the threads that run the doall; none: all of them."
    )
    container: str | None = pydantic.Field(
        None,
        description="sink-invariant-code: the temporary whose statements move into the loop or doall, as list_moves "
        "names it (the state's writes=[..] shows it).",
    )
    fp_mode: str | None = pydantic.Field(
        None, description="set-fp-mode (labels [], scope: the kernel): strict-ieee, contract-fma or fast-math."
    )
    dtype: str | None = pydantic.Field(
        None,
        description="change-precision (labels [], container): the element type the array takes, float32 or float64.",
    )
    at: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description='split-doall: the parameter to cut and where, e.g. {"i": "2"} or {"i": "N - 1"}; the doall over '
        "[lo, at) stays, the doall over [at, hi] follows in a new state.",
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
    scope: str | None = pydantic.Field(
        None,
        description="The kernel whose body holds the rows, as list_kernels names it; none for run-on-cpu, run-on-gpu and "
        "their -persistent forms, whose labels are the kernels.",
    )


class ListScopeMoves(ListMoves, InScope):
    pass


#: The value each optional ``apply_move`` argument takes when not given; a listed move prints only the others.
MOVE_DEFAULTS: dict[str, object] = {
    name: field.get_default(call_default_factory=True)
    for name, field in ApplyMove.model_fields.items()
    if not field.is_required()
}


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


class GotoPhase(Args):
    phase: str = pydantic.Field(
        description="An earlier phase to return to: structured, scopes, scoped or kernels (one the run gives the agent)."
    )


class ToolDef(NamedTuple):
    phases: frozenset[Phase]
    args: type[Args]
    description: str
    effect: Effect = Effect.ADD

    @property
    def label(self) -> str:
        """The phases as the tool's description and its record name them: ``structured|scoped``, or ``any``."""
        return "|".join(p.value for p in Phase if p in self.phases)


def phases(*names: Phase) -> frozenset[Phase]:
    return frozenset(names)


TOOLS: dict[str, ToolDef] = {
    "advance_phase": ToolDef(
        phases(Phase.ANY),
        NoArgs,
        "Leave this phase for the next one; the phases between run their defaults. The answer names the new phase and "
        "what ran on the way.",
    ),
    "undo_phase": ToolDef(
        phases(Phase.ANY),
        NoArgs,
        "Take this phase back completely, with the defaults run on the way into it, and return to the previous phase as "
        "you left it.",
        Effect.REPLACE,
    ),
    "return_to_phase": ToolDef(
        phases(Phase.ANY),
        GotoPhase,
        "Return to an earlier phase, taking every phase after it back (undo_phase until there); e.g. after a score.",
        Effect.REPLACE,
    ),
    "get_cpf": ToolDef(
        phases(Phase.ANY),
        NoArgs,
        "Start here: the program in canonical parallel form, as a labeled tree.",
        Effect.READ,
    ),
    "view_tree": ToolDef(
        phases(Phase.ANY),
        ViewTree,
        "Open states and doalls with everything below them, close some again, print bodies as C or Python, then show "
        "the tree, or one row's subtree (root). Changes only what is shown, never the program.",
        Effect.READ,
    ),
    "get_work_depth": ToolDef(
        phases(Phase.ANY),
        GetWorkDepth,
        "Work (operations) and depth (the longest dependent chain) of tree rows, at this session's sizes or as "
        "formulas: rank loops by them before choosing what to move. The tree does not print them.",
        Effect.READ,
    ),
    "list_move_kinds": ToolDef(
        phases(Phase.ANY),
        NoArgs,
        "Every move kind: the tree rows it takes, the transformation it runs, what it does.",
        Effect.READ,
    ),
    "list_moves": ToolDef(
        phases(Phase.STRUCTURED, Phase.SCOPED),
        ListMoves,
        "Every legal move right now: fusion, fission, interchange, tiling, unrolling, library expansion and "
        "implementation, and propagate-config.",
        Effect.READ,
    ),
    "apply_moves": ToolDef(
        phases(Phase.STRUCTURED, Phase.SCOPED),
        ApplyMoves,
        "Apply several moves in order, all or nothing: after each move the program is built and run at the small "
        "check sizes against the reference. The first move refused or found wrong takes the whole batch back; the "
        "answer names it (failed_at), the substep (apply or check) and why. Labels change as moves apply: name each "
        "move's rows as the tree has them when it runs.",
    ),
    "apply_move": ToolDef(
        phases(Phase.STRUCTURED, Phase.SCOPED),
        ApplyMove,
        "Apply one move as list_moves listed it (or with your own tile_sizes, lanes, block, max_iterations, config_values); answers "
        "at once with the outcome, not with a time: call get_performance after the moves you want to compare. It "
        "retires the labels of the rows it took; its tree field is the subtree it rewrote under the rows around it (the "
        "rest of the tree you read is unchanged), or the whole tree when the change reaches the top. kind rollback, labels [], "
        "takes the last change back (the last 8 can be).",
    ),
    "tune_move": ToolDef(
        phases(Phase.STRUCTURED, Phase.SCOPED),
        TuneMove,
        "Try a tiling, vectorize-doall or set-schedule-gpu move at several sizes in one call, each measured as "
        "get_performance measures. The fastest "
        "size that beats the program now is kept (status kept); if none does, the program stays as it was. Use it "
        "instead of tiling, untiling and tiling again.",
    ),
    "set_llvm_passes": ToolDef(
        phases(Phase.ANY),
        SetLlvmPasses,
        "Build every kernel through this LLVM pass pipeline (opt -passes= syntax): its C++ is lowered to IR without "
        "optimization, the pipeline runs, then only the back end. The syntax is checked now; get_performance checks "
        "the results. A second call replaces the pipeline.",
    ),
    "set_mlir_passes": ToolDef(
        phases(Phase.ANY),
        SetMlirPasses,
        "Build every kernel through this MLIR pass pipeline (polygeist-opt --pass-pipeline= syntax) on the affine MLIR "
        "cgeist makes of its C++, then a fixed lowering to LLVM. The syntax is checked now; get_performance checks the "
        "results. A second call replaces the pipeline.",
    ),
    "request_transformation": ToolDef(
        phases(Phase.ANY),
        RequestTransformation,
        "If you think you need a transformation no move offers (list_moves on the rows found none), ask for it: "
        "explain what it does and when it is legal, and give at least 5 pairs of small programs of your own before and "
        "after it, in Python (one @dace.program each) or C, computing the same results. The request is checked and "
        "kept for a later round; it does not change this program.",
        Effect.READ,
    ),
    "run_defaults": ToolDef(
        phases(Phase.STRUCTURED, Phase.SCOPES),
        RunDefaults,
        "Finish one stage with NestForge's default heuristic; call get_performance to time it.",
    ),
    "set_scope": ToolDef(
        phases(Phase.SCOPES),
        Labeled,
        "Make one kernel of several top-level doalls of one state, or of a straight run of blocks; answers with the "
        "kernel's name.",
    ),
    "unset_scope": ToolDef(phases(Phase.SCOPES), UnsetScope, "Inline a kernel back into the program.", Effect.REPLACE),
    "list_scope_moves": ToolDef(
        phases(Phase.SCOPES),
        ListScopeMoves,
        "The moves of the scopes phase: run-on-cpu, run-on-gpu and their -persistent forms on every kernel and every "
        "run of consecutive kernels; with scope, the schedules and persistent runs inside that kernel.",
        Effect.READ,
    ),
    "apply_scope_move": ToolDef(
        phases(Phase.SCOPES),
        ApplyScopeMove,
        "A move of the scopes phase, as list_scope_moves listed it: run-on-* places kernels (rollback takes it back); "
        "a schedule or persistent run names its kernel (scope). A placed kernel is placed again with the change.",
    ),
    "finalize_scope": ToolDef(
        phases(Phase.SCOPES),
        NoArgs,
        "Finish the scopes: what no run-on move placed gets its default device, and the GPU kernels' host/device "
        "copies move into the program, so a value that stays on the GPU between kernels is copied once. Answers each "
        "kernel's device and the copies; measure (get_performance) after it, not before.",
    ),
    "list_kernels": ToolDef(phases(Phase.KERNELS), NoArgs, "Every kernel with its device and arguments.", Effect.READ),
    "focus_kernel": ToolDef(
        phases(Phase.KERNELS),
        KernelName,
        "Scope your kernel work to one kernel: write_kernel and tune_kernel take only the kernel in focus.",
    ),
    "read_kernel": ToolDef(phases(Phase.KERNELS), KernelName, "A kernel's current source.", Effect.READ),
    "write_kernel": ToolDef(
        phases(Phase.KERNELS),
        WriteKernel,
        "Replace a kernel's source; it is built, checked against the kernel's oracle and linked only when it matches "
        "(status unchecked: it has no oracle, so it is linked and the whole program checks it).",
        Effect.REPLACE,
    ),
    "tune_kernel": ToolDef(phases(Phase.KERNELS), TuneKernel, "Sweep compilers and flags for one kernel."),
    "get_performance": ToolDef(
        phases(Phase.ANY),
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
        phases(Phase.ANY),
        NoArgs,
        "Every change this program holds, in order: moves, defaults, scopes, kernels.",
        Effect.READ,
    ),
    "finalize": ToolDef(phases(Phase.ANY), NoArgs, "Build the program as one library in the bench C ABI."),
}

ApplyMoves.model_rebuild()
