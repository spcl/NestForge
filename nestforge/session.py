# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the one API over the seven stages, shared by the deterministic driver, humans and agents.

Callers name tree rows by their labels and kernels by name. A label stays on its row while the row lives; a change
retires the labels of the rows it consumed and names what it made afresh, so a label read before it either still
names the same row or is refused as unknown, never acting on a moved graph. Every stage has a deterministic default; the agent calls are the alternatives to it.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import dataclasses
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, Literal, cast

import dace
import numpy as np
import pydantic
from dace.libraries.standard.nodes.external_call import ExternalCall, data_name, external_calls
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from relayout.ops import Operation

from nestforge.build.arena import TIMED_REPS
from nestforge.build.sdfg import PERSISTENT_MIN_BYTES, force_register_arrays, make_persistent
from nestforge.build.toolchain import SymbolValue, parse_params, raw_signature
from nestforge.corpus.translate import Prepared, prepare
from nestforge.ir.dace_types import strings
from nestforge.ir.depends import KernelGraph, UnsupportedProgram, kernel_dependencies
from nestforge.ir.extract import find_state_of_node
from nestforge.ir.introspect import (
    PYTHON,
    WORK_DEPTH_FAILURES,
    C,
    Language,
    Row,
    WorkDepth,
    describe_graph,
    enclosing_rows,
    guarded_work_depth,
    tree_rows,
)
from nestforge.ir.loops import loop_symbol_values
from nestforge.ir.names import Namer, inline_top_level_nsdfgs, normalize_labels
from nestforge.records import (
    CRASHES,
    KernelOutcomeStatus,
    KernelStatus,
    MoveOutcome,
    MoveStatus,
    Record,
    Refusal,
    RefusalOutcome,
    Refused,
    UnclassedRefusal,
    refusal_of,
    spelled,
)
from nestforge.stages import feedback
from nestforge.stages.canonicalize import (
    Targets,
    canonicalize,
    explicit_copies,
    finish,
    fuse_and_finish,
    implicit_copies,
    narrow_nested_memlets,
    skew_wavefronts,
    structural_cleanup,
)
from nestforge.stages.expand import implementations, set_default_implementations, settle
from nestforge.stages.families import DEFAULT_MOVES_FAMILY, WAVEFRONT_FAMILY, checked_families, switched_off
from nestforge.stages.finalize import BenchTarget, Final, assemble
from nestforge.stages.kernel import (
    DEVICE_UNIT_SUFFIX,
    KernelSource,
    KernelVerdict,
    Probe,
    bounded_probe,
    build_kernel_library,
    computed_symbols_reason,
    full_probe,
    kernel_form,
    kernel_runtime_libraries,
    kernel_text,
    measure_kernel,
    schedule_kernel,
    split_kernel_text,
    unchecked_verdict,
    use_kernel_library,
)
from nestforge.stages.layout import (
    RELAYOUT,
    argument_layouts,
    laid_out,
    layout_operation,
    layout_refusal,
    listed_operation,
    physical_inputs,
    requests_of,
)
from nestforge.stages.moves import (
    DEFAULT_ISA,
    DEVICE_KINDS,
    EXPAND_LIBRARY,
    FORCED_LOOP_TO_MAP,
    MOVE_SHAPES,
    PROBE_UNROLL,
    PROPAGATE_CONFIG,
    ROLLBACK,
    SCOPE_KINDS,
    SET_IMPLEMENTATION,
    SET_SCHEDULE_CPU,
    SET_SCHEDULE_GPU,
    SIZED,
    GpuChoice,
    Rewrite,
    check_kind,
    legal_moves,
    plan_move,
    plan_set_schedule_gpu,
    probe_sizes,
    schedule_text,
    scope_metrics,
    sdfg_metrics,
    set_gpu_choices,
)
from nestforge.stages.performance import Performance, Reference, kernel_reports, offload_report, time_program
from nestforge.stages.placement import (
    Placement,
    Transfer,
    check_devices,
    copies_in_call,
    default_devices,
    has_device_memory,
    insert_copies,
    kernel_device,
    kernel_interfaces,
    place,
    transfers,
)
from nestforge.stages.scopes import (
    is_parallel_kernel,
    lower_default_scopes,
    lower_group_to_external_call,
    node_boundary,
)
from nestforge.stages.variants import Variant, VariantCell, device_variants, select_variant
from nestforge.stages.vectorize import AUTO_VECTORIZE

#: Characters of a failed build's error text a caller gets back.
ERROR_TAIL = 1500

#: How many of the last moves and default move stages :meth:`Session.undo` can take back, each a copy of the program.
UNDO_DEPTH = 8


#: The kernel source language each device's kernels are written in.
def kernel_language(device: str) -> str:
    """The ``write_kernel`` language of a kernel on ``device``: ``cpp`` on the CPU, the GPU form's dialect on the GPU
    (``cuda`` or ``hip``)."""
    return "cpp" if device == "cpu" else kernel_form(device).language


class Stopped(BaseException):
    """A stopped call's work ends here: not an ``Exception``, so no refused build or failed run absorbs it."""


class Stopper:
    """Stops long measuring work between signals of the caller. Only an armed section stops at once, in the middle of a
    build or a run; it holds no change to the program, so stopping there leaves the session whole."""

    __slots__ = ("armed_now", "asked")

    def __init__(self) -> None:
        self.armed_now = False
        self.asked = False

    def ask(self) -> None:
        """Stop the call: at once in an armed section, else on entering the next one."""
        self.asked = True
        if self.armed_now:
            self.armed_now = False  # the caller signals until it hears back: cleanup on the way out runs to its end
            raise Stopped

    def clear(self) -> None:
        self.asked = False

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        before, self.armed_now = self.armed_now, True
        try:
            if self.asked:
                raise Stopped
            yield
        finally:
            self.armed_now = before and not self.asked


def computed_symbols(src: KernelSource, sizes: Mapping[str, SymbolValue]) -> list[str]:
    """The symbols kernel ``src`` takes that ``sizes`` has no value for: ones only the running program computes."""
    return [s for s in src.boundary.symbols if s not in sizes]


#: What a forced move tells the agent.
FORCED_WARNING = (
    "forced: the compiler did not prove this move legal. If the iterations depend on each other, the program is "
    "wrong, and only a run on data shows it; rollback takes the move back."
)
#: The kind a default stage records in ``Session.applied``, before the stage's name.
DEFAULTS_PREFIX = "defaults-"


class MoveResult(Record):
    """What a labeled call did; ``reason`` names what ran once ``applied``, else why nothing changed."""

    status: MoveOutcome
    kind: str
    labels: tuple[str, ...]
    reason: str
    #: The kernel whose body the move rewrote; ``None`` for the program.
    scope: str | None = None
    #: Why nothing changed, as a class (:class:`~nestforge.records.Refusal`); ``None`` once ``applied``.
    refusal: RefusalOutcome | None = None
    #: Whether the move was applied with ``force``, past a check that refused it.
    forced: bool = False
    #: What the agent should know about an applied move; ``None`` without anything to say.
    warning: str | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def classed(cls, data: Any) -> Any:
        """A refused result takes the class its reason carries, unless one is given."""
        if not isinstance(data, dict) or data.get("status") is MoveStatus.APPLIED or data.get("refusal") is not None:
            return data
        refusal = refusal_of(data.get("reason", ""))
        if refusal is Refusal.OTHER:
            warnings.warn(UnclassedRefusal(f"{data.get('kind')}: {data.get('reason')}"), stacklevel=2)
        return {**data, "refusal": refusal}


class Move(Record):
    """A legal move, exactly as :meth:`Session.apply_move` takes it back."""

    kind: str
    labels: tuple[str, ...]
    #: For a sized kind (``tile-doall``, ``tile-for``), the sizes it was checked at; the agent may pass its own.
    tile_sizes: tuple[int, ...] = ()
    #: For ``set-library-implementation``, the implementation it sets, and the one the call has now; for
    #: ``set-schedule-cpu``, ``current`` is the doall's schedule now (``dynamic,16 threads=8``).
    implementation: str | None = None
    current: str | None = None
    #: For ``unroll-for`` and ``unroll-doall``, the cap it was checked at; the agent may pass its own.
    max_iterations: int = PROBE_UNROLL
    #: For ``propagate-config``, the values it specializes for; listed without any, the agent passes them.
    config_values: dict[str, int | float] = {}
    #: For ``relayout``, the operation it applies; listed with :func:`~nestforge.stages.layout.listed_operation`.
    operation: dict[str, object] = {}
    #: For ``auto-vectorize-scope``, the ISA its tiles are for; ``tile_sizes`` holds their lanes.
    isa: str = DEFAULT_ISA
    #: For ``for-to-doall``, whether it applies past a refused dependence check; a listed move never does.
    force: bool = False
    #: For ``set-schedule-cpu``, the OpenMP schedule (``static``, ``dynamic``, ``guided``, ``default``), its chunk,
    #: and the team size; ``None``: no clause.
    schedule: str | None = None
    chunk: int | None = None
    threads: int | None = None
    #: The kernel whose body the move rewrites; ``None`` for the program.
    scope: str | None = None


class KernelScope(Record):
    """A kernel :meth:`Session.default_scopes` made, with the containers it reads and writes."""

    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    symbols: tuple[str, ...]
    parallel: bool


class KernelInfo(Record):
    """A kernel, its device and arguments, the producer labels reaching each argument (``depends``) and the loops a
    reaching value crossed (``carried``)."""

    name: str
    device: str
    parallel: bool
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    symbols: tuple[str, ...]
    depends: dict[str, list[str]]
    carried: dict[str, list[str]]


def kernel_named(sdfg: dace.SDFG, name: str) -> ExternalCall:
    return next(ext for ext in external_calls(sdfg) if ext.name == name)


class PlaceResult(Record):
    """Once ``applied``: every kernel's device, the copies placed and the transfers the kernel DAG implies
    (``None`` when the analysis refuses the program)."""

    status: MoveOutcome
    reason: str = ""
    devices: dict[str, str] = {}
    copies: tuple[tuple[str, str], ...] = ()
    transfers: tuple[Transfer, ...] | None = None


class KernelLibrary(Record):
    """What :meth:`Session.optimize_kernel` built and linked."""

    kernel: str
    symbol: str
    abi_order: tuple[str, ...]
    unit: Path
    library: Path
    variant: str

    @property
    def entry(self) -> str:
        return f"{self.symbol}({', '.join(self.abi_order)})"


class KernelOutcome(Record):
    """What :meth:`Session.set_kernel_source` did."""

    kernel: str
    status: KernelOutcomeStatus
    reason: str = ""
    time_us: float | None = None


class Config(Record):
    """The configuration stage 6 chose."""

    compiler: str
    fp_mode: str
    cost_model: str
    flags: tuple[str, ...]
    time_us: float


class Sweep(Record):
    """Stage 6 over one kernel: how many variants were measured, and the winner."""

    kernel: str
    cells: int
    winner: str | None
    config: Config | None


@dataclass(slots=True)
class KernelBuild:
    """The library a kernel links now: the unit it was built from, the variant that built it, its last verdict and
    the compiler's remarks on it."""

    source: KernelSource
    variant: Variant
    archive: Path
    verdict: KernelVerdict | None = None
    remarks: str | None = None


#: Why a rebuild's repeat links a kernel unchecked.
REPEATED = "checked when the agent made it; repeated after the session restarted"
#: How :meth:`Session.default_build` builds a kernel: stage 5's first configuration or stage 6's sweep winner.
Codegen = Literal["first", "sweep"]
#: What :meth:`Session.run_defaults` finishes.
Stage = Literal["moves", "finish", "scopes", "placement", "round"]
#: A measurement and what it was of: the inputs, the reference, the repetitions and the codegen.
Measured = tuple[dict[str, np.ndarray], Reference | None, int, Codegen, Performance]
#: A sweep winner's identity: the kernel, the text of its units, and the sizes it was timed at.
SweepKey = tuple[str, tuple[str, ...], tuple[tuple[str, SymbolValue], ...]]


def kept(build: KernelBuild, folder: Path) -> KernelBuild:
    """``build`` with its units, archive and shared twin copied into ``folder``, so it outlives the work directory it
    was built in."""
    folder.mkdir(parents=True, exist_ok=True)
    shared = build.archive.with_suffix(".so")
    paths = [*build.source.units, build.archive, *([shared] if shared.exists() else [])]
    copies = {path: Path(shutil.copy2(path, folder / path.name)) for path in paths}
    device = build.source.device_unit
    source = dataclasses.replace(
        build.source, unit=copies[build.source.unit], device_unit=copies[device] if device is not None else None
    )
    return KernelBuild(source, build.variant, copies[build.archive], build.verdict)


def scratch_dir() -> Path:
    """A work directory of the session's own, removed when the process exits."""
    path = Path(tempfile.mkdtemp(prefix="nfsession_"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def undone_on_failure[**Params, Answer](
    change: Callable[Concatenate[Session, Params], Answer],
) -> Callable[Concatenate[Session, Params], Answer]:
    """``change``, a :class:`Session` method, under :meth:`Session.atomic`."""

    @wraps(change)
    def guarded(session: Session, *args: Params.args, **kwargs: Params.kwargs) -> Answer:
        with session.atomic():
            return change(session, *args, **kwargs)

    return guarded


@dataclass(slots=True)
class Checkpoint:
    """What a change to the program can touch: :meth:`Session.atomic` puts it back when the change fails."""

    sdfg: dace.SDFG
    unplaced: dace.SDFG | None
    unscoped: tuple[dace.SDFG, Namer, list[MoveResult]] | None
    unlaid: dace.SDFG | None
    layout: tuple[Operation, ...]
    namer: Namer
    applied: list[MoveResult]
    expanded: dict[str, bool]
    languages: dict[str, Language]
    costs: WorkDepth | None
    costs_skipped: str | None
    sources: dict[str, KernelSource]
    builds: dict[str, KernelBuild]
    prepared: dict[str, Prepared]
    measured: Measured | None
    gpu_choices: dict[str, dict[str, GpuChoice]]


class Session:
    """Owner of one program SDFG, driven stage by stage."""

    __slots__ = (
        "applied",
        "shared_root",
        "sdfg",
        "name",
        "targets",
        "sizes",
        "namer",
        "rows",
        "expanded",
        "languages",
        "costs",
        "costs_skipped",
        "work_dir",
        "kernel_deps",
        "unplaced",
        "unscoped",
        "unlaid",
        "layout",
        "sources",
        "builds",
        "prepared",
        "codegen",
        "swept",
        "swept_dir",
        "first_builds",
        "sweep_cells",
        "check_sizes",
        "oracles_out_of_time",
        "measured",
        "gpu_choices",
        "bench",
        "stopper",
        "in_atomic",
        "checkpoint",
        "history",
        "focus",
        "disabled_families",
    )

    def __init__(
        self,
        sdfg: dace.SDFG,
        targets: Targets | None = None,
        name: str | None = None,
        work_dir: str | None = None,
        sizes: Mapping[str, SymbolValue] | None = None,
        codegen: Codegen = "first",
        namer: Namer | None = None,
        disabled_families: Iterable[str] = (),
    ) -> None:
        """:param sizes: Value of every symbol, used to validate and time kernels (stages 5 to 7).
        :param codegen: How a kernel without a build of its own is built when the program is timed or finalized:
            ``first``, stage 5's first configuration; ``sweep``, the fastest correct one of stage 6's sweep.
        :param namer: The row names ``sdfg`` already carries (a copy of another session's); fresh by default.
        :param disabled_families: Move families (:data:`~nestforge.stages.families.FAMILIES`, or ``all``) this run
            switches off: no listing offers their kinds, no call applies one, and no default runs one.
        :raises ValueError: A name in ``disabled_families`` is no family."""
        #: The move families switched off for the whole run.
        self.disabled_families = checked_families(disabled_families)
        self.sdfg = sdfg
        self.targets = targets if targets is not None else Targets()
        self.name = name or sdfg.label
        self.sizes: dict[str, SymbolValue] = dict(sizes) if sizes is not None else {}
        self.namer = namer if namer is not None else Namer()
        normalize_labels(sdfg, self.namer)
        self.rows: dict[str, Row] | None = None
        #: Every call that changed the program, in order: moves, defaults, scopes and written kernels.
        self.applied: list[MoveResult] = []
        #: The folder under which finalize builds a library the judge accepts; the shared folder when ``None``.
        self.shared_root: str | None = None
        # labels whose content describe() shows, kept while their rows live, and the work and depth it prints
        self.expanded: dict[str, bool] = {}
        # labels whose leaf bodies describe() prints in another language than the row around them, kept while they live
        self.languages: dict[str, Language] = {}
        self.costs: WorkDepth | None = None
        # why describe() printed no costs for this program, or None when it printed them
        self.costs_skipped: str | None = None
        self.work_dir = Path(work_dir) if work_dir else scratch_dir()
        self.kernel_deps: KernelGraph | None = None
        # the program before stage 4, so a new placement starts from it
        self.unplaced: dace.SDFG | None = None
        # the program before its first kernel, its row names and what it holds, so a new round starts from it
        self.unscoped: tuple[dace.SDFG, Namer, list[MoveResult]] | None = None
        # the program before its first relayout, which every relayout lays out again with all the operations so far
        self.unlaid: dace.SDFG | None = None
        #: The relayout operations the program works on its arguments by, in order; the bench's ``requests``.
        self.layout: tuple[Operation, ...] = ()
        self.sources: dict[str, KernelSource] = {}
        self.builds: dict[str, KernelBuild] = {}
        self.prepared: dict[str, Prepared] = {}
        self.codegen = codegen
        # sweep winners by kernel source and sizes, shared with every trial so a kernel is swept once
        self.swept: dict[SweepKey, KernelBuild] = {}
        # where a cached winner's units and archive are kept, so they outlive the trial that swept them
        self.swept_dir = self.work_dir / "swept"
        # first-configuration builds by the same key, with their verdicts: a move rebuilds only the kernels it changed
        self.first_builds: dict[SweepKey, KernelBuild] = {}
        # the cells of a stopped sweep, by the same key
        self.sweep_cells: dict[SweepKey, dict[Variant, VariantCell]] = {}
        # the sizes a kernel's oracle falls back to when it cannot finish at ``sizes``; None: no bound on the oracle
        self.check_sizes: dict[str, SymbolValue] | None = None
        # the oracles (by source) that did not finish at ``sizes``, so a retry does not spend their budget again
        self.oracles_out_of_time: set[str] = set()
        # :meth:`performance` of the current program, dropped by every change to it
        self.measured: Measured | None = None
        #: Per GPU kernel, what set-schedule-gpu chose for its doalls, by label; set again after every placement.
        self.gpu_choices: dict[str, dict[str, GpuChoice]] = {}
        # the bench task :meth:`finalize` assembles for; set by whoever started the session from a bench kernel
        self.bench: BenchTarget | None = None
        self.stopper = Stopper()
        # inside :meth:`atomic`: its outermost call owns the checkpoint
        self.in_atomic = False
        # the program before the change :meth:`atomic` is running, and before each of the last changes kept for undo
        self.checkpoint: Checkpoint | None = None
        self.history: list[Checkpoint] = []
        #: The kernel the agent's kernel work is scoped to (:meth:`focus_kernel`); ``None`` before the first.
        self.focus: str | None = None

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """A failure or a stop inside the block leaves the session as it was: a rewrite in place cannot be undone, so the
        block runs on the program while a copy waits to replace it. Calls nest; the outermost restores."""
        if self.in_atomic:
            yield
            return
        saved = Checkpoint(
            copy.deepcopy(self.sdfg),
            copy.deepcopy(self.unplaced),
            self.unscoped,  # replaced, never changed
            self.unlaid,  # replaced, never changed
            self.layout,
            copy.deepcopy(self.namer),
            list(self.applied),
            dict(self.expanded),
            dict(self.languages),
            self.costs,
            self.costs_skipped,
            dict(self.sources),
            dict(self.builds),
            dict(self.prepared),
            self.measured,
            {name: dict(held) for name, held in self.gpu_choices.items()},
        )
        kept = len(self.history)
        self.in_atomic, self.checkpoint = True, saved
        try:
            yield
        except (*CRASHES, Stopped):
            self.restore(saved)
            del self.history[kept:]
            raise
        finally:
            self.in_atomic, self.checkpoint = False, None

    def restore(self, saved: Checkpoint) -> None:
        """Make ``saved`` the session's program again."""
        self.sdfg, self.unplaced, self.namer = saved.sdfg, saved.unplaced, saved.namer
        self.unscoped = saved.unscoped
        self.unlaid, self.layout = saved.unlaid, saved.layout
        self.applied, self.expanded, self.languages = saved.applied, saved.expanded, saved.languages
        self.costs, self.costs_skipped = saved.costs, saved.costs_skipped
        # the rows and the kernel graph hold nodes of the program just dropped
        self.rows, self.kernel_deps = None, None
        self.sources, self.builds, self.prepared = saved.sources, saved.builds, saved.prepared
        self.measured = saved.measured
        self.gpu_choices = saved.gpu_choices

    def remember(self) -> None:
        """Keep the program as it was before the change :meth:`atomic` is running, for :meth:`undo`; the last
        :data:`UNDO_DEPTH` changes are kept."""
        assert self.checkpoint is not None, "a change runs under atomic"
        self.history.append(self.checkpoint)
        del self.history[:-UNDO_DEPTH]

    @undone_on_failure
    def undo(self) -> MoveResult:
        """Back to the program before the last move or default stage :meth:`remember` kept, its row labels and
        :attr:`applied` included; a label handed out since is never handed out again. Always legal: with nothing kept, it
        changes nothing."""
        if not self.history:
            reason = f"nothing to roll back: no change was made, or the last {UNDO_DEPTH} are taken back; unchanged."
            return MoveResult(status=MoveStatus.APPLIED, kind=ROLLBACK, labels=(), reason=reason)
        saved, counters = self.history.pop(), self.namer.counters
        undone = self.applied[min(len(saved.applied), len(self.applied) - 1)]
        self.restore(saved)
        self.namer.counters = {kind: max(n, saved.namer.counters.get(kind, 0)) for kind, n in counters.items()}
        return MoveResult(status=MoveStatus.APPLIED, kind=ROLLBACK, labels=undone.labels, reason=f"undid {undone.kind}")

    def bump(self, consumed: Sequence[str] = ()) -> None:
        """After a change to the program: retire the labels of the rows it ``consumed`` and name its new rows."""
        self.namer.retire(tuple(consumed))
        normalize_labels(self.sdfg, self.namer)
        set_default_implementations(self.sdfg, self.targets)
        # placement decides a kernel's device: pinned to the host stack now, a GPU kernel's arrays would stay there
        force_register_arrays(self.sdfg, kernels=not self.targets.gpu)
        self.rows = None
        self.expanded = {label: is_open for label, is_open in self.expanded.items() if label in self.row_index()}
        self.languages = {label: lang for label, lang in self.languages.items() if label in self.row_index()}
        self.costs = None
        self.costs_skipped = None
        self.kernel_deps = None
        self.sources, self.builds, self.prepared = {}, {}, {}
        self.measured = None

    def row_index(self) -> dict[str, Row]:
        """Tree label -> ``(block or node, state)``, built once per change of the program."""
        if self.rows is None:
            self.rows = tree_rows(self.sdfg)
        return self.rows

    def named_rows(
        self, kind: str, names: tuple[str, ...], rows: dict[str, Row] | None = None
    ) -> list[Row] | MoveResult:
        """The rows ``names`` label (in ``rows``, the program's by default), or the ``not-found`` result: a label no
        row holds, or one a change retired."""
        rows = self.row_index() if rows is None else rows
        missing = [name for name in names if name not in rows]
        if missing:
            gone = [name for name in missing if name in self.namer.retired]
            why = f" ({', '.join(gone)} changed; read the tree again)" if gone else ""
            return MoveResult(
                status=MoveStatus.NOT_FOUND,
                kind=kind,
                labels=names,
                reason=Refused(f"no tree row is labeled {', '.join(missing)}{why}.", Refusal.UNKNOWN_LABEL),
            )
        return [rows[name] for name in names]

    def move_programs(self, scope: str | None) -> tuple[dace.SDFG, dace.SDFG]:
        """``(what the structured moves rewrite, what the device moves rewrite)``: the program, or the body of the
        kernel ``scope`` before placement and as placement lowered it."""
        if scope is None:
            return self.sdfg, self.sdfg
        lowered = self.kernel(scope).standalone_sdfg
        source = lowered if self.unplaced is None else kernel_named(self.unplaced, scope).standalone_sdfg
        if source is None or lowered is None:
            raise ValueError(f"{scope} keeps no SDFG of its nest to rewrite.")
        return source, lowered

    # Stage 1: canonicalize

    def canonicalize(self) -> str:
        """Canonicalize up to the fusion stage for the session's targets; returns the new tree."""
        canonicalize(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    # Stage 2: moves

    def describe(self, bodies: bool = False, deps: bool = False) -> str:
        """The program as a text tree; rows are named by the labels the labeled calls take.
        Every loop and state is a row, loop and map rows carry work and depth, and a state shows its kernels only
        once :meth:`expand` opened it. ``deps`` puts each kernel's :meth:`kernel_graph` line under its row."""
        if self.costs is None:
            self.costs, self.costs_skipped = guarded_work_depth(self.sdfg)
        tree = describe_graph(
            self.sdfg,
            bodies=bodies,
            notes=self.deps_line if deps else None,
            expanded=self.expanded,
            costs=self.costs,
            languages=self.languages,
        )
        if self.costs_skipped is None:
            return tree
        header, _, rows = tree.partition("\n")
        return f"{header}\n(no work= or depth=: {self.costs_skipped})\n{rows}"

    def expand(self, labels: Sequence[str]) -> str:
        """Show everything below the states and maps ``labels`` name (kernels, nested maps and leaf bodies, a map's
        nested states, all opened) until the program changes; returns what opened, or why nothing did."""
        rows = self.named_rows("expand", tuple(labels))
        if isinstance(rows, MoveResult):
            return f"{spelled(rows.status)}: {rows.reason}"
        closed = [name for name, (obj, _) in zip(labels, rows) if not isinstance(obj, (SDFGState, nodes.MapEntry))]
        if closed:
            return f"illegal: only states and maps expand; {', '.join(closed)} is neither."
        for label in labels:
            # opened whole: what was closed inside it opens with it
            inner = [row for row, (obj, state) in self.row_index().items() if label in enclosing_rows(obj, state)]
            for row in inner:
                self.expanded.pop(row, None)
            self.expanded[label] = True
        return f"ok: expanded {', '.join(labels)}"

    def is_open(self, label: str) -> bool:
        """Whether the row ``label`` shows its content: its own view entry, else that of the nearest row around it."""
        obj, state = self.row_index()[label]
        for row in (label, *enclosing_rows(obj, state)):
            if row in self.expanded:
                return self.expanded[row]
        return False

    def view(
        self,
        expand: Sequence[str] = (),
        collapse: Sequence[str] = (),
        c: Sequence[str] = (),
        python: Sequence[str] = (),
    ) -> str:
        """:meth:`expand`, :meth:`collapse`, then :meth:`set_language` for ``c`` and ``python``, so one call opens a
        row, closes some inside it and picks how bodies print; the tree after them, each note first. Only what is
        shown changes, never the program."""
        notes = [self.expand(expand)] if expand else []
        if collapse:
            notes.append(self.collapse(collapse))
        notes += [self.set_language(labels, lang) for labels, lang in ((c, C), (python, PYTHON)) if labels]
        return "\n".join([*notes, self.describe()])

    def set_language(self, labels: Sequence[str], language: Language) -> str:
        """Print the leaf bodies below the rows ``labels`` name in ``language`` (Python by default; C shows C++
        tasklets, intrinsics and inline assembly, as written), until they retire; a row inside one keeps its own
        choice. Returns what changed, or why nothing did."""
        rows = self.row_index()
        unknown = [label for label in labels if label not in rows]
        if unknown:
            return f"illegal: no tree row is labeled {', '.join(unknown)}; nothing changed."
        for label in labels:
            self.languages[label] = language
        return f"ok: {', '.join(labels)} print bodies in {language}"

    @undone_on_failure
    def run_defaults(self, stage: Stage) -> object:
        """Finish ``stage`` with its default: :meth:`default_moves`, :meth:`finish_moves`, :meth:`default_scopes`,
        :meth:`default_placement` after the default scopes of what no scope covers, or :meth:`rewind`. A default that
        changed the program is on :attr:`applied`, which a round takes back to what the program it restores holds."""
        done: object
        changed: object
        match stage:
            case "moves":
                done = changed = self.default_moves()
            case "finish":
                done = changed = self.finish_moves()
            case "scopes":
                self.keep_unscoped()
                done = changed = self.default_scopes()
            case "placement":
                self.keep_unscoped()
                scoped = self.default_scopes() if self.unplaced is None else []
                placed = self.default_placement()
                changed = scoped or placed is not None
                done = placed or PlaceResult(status=MoveStatus.ILLEGAL, reason="the kernels are placed already.")
            case "round":
                return self.rewind()
        if changed:
            self.applied.append(
                MoveResult(status=MoveStatus.APPLIED, kind=f"{DEFAULTS_PREFIX}{stage}", labels=(), reason="")
            )
            if stage in ("moves", "finish"):
                self.remember()
        return done

    def collapse(self, labels: Sequence[str]) -> str:
        """Hide the content of the open states and maps ``labels`` name, also one opened with a row around it; returns
        what closed, and which labels were not open."""
        rows = self.row_index()
        closed = [label for label in labels if label in rows and self.is_open(label)]
        shut = [label for label in labels if label not in closed]
        for label in closed:
            self.expanded[label] = False
        notes = [f"ok: collapsed {', '.join(closed)}"] if closed else []
        if shut:
            notes.append(f"not open: {', '.join(shut)}; only an open state or map collapses.")
        return " ".join(notes)

    def deps_line(self, node: nodes.LibraryNode) -> str | None:
        graph = self.kernel_graph()
        return graph.line(node.label) if node.label in graph.kernels else None

    def list_moves(self, kind: str | None = None, scope: str | None = None) -> list[Move]:
        """Every legal move right now, of ``kind`` or of every kind, in the program or inside the kernel ``scope``."""
        structured, device = self.move_programs(scope)
        target = self.targets.canon_target
        found = [move for move in legal_moves(structured, kind, target) if move[0] not in DEVICE_KINDS]
        found += [move for k in sorted(DEVICE_KINDS) if kind in (None, k) for move in legal_moves(device, k, target)]
        rows = self.row_index() if scope is None else tree_rows(structured)
        # set only inside a kernel: an answer carries the fields set, and apply_move takes no scope
        scoped: dict[str, str] = {} if scope is None else {"scope": scope}
        moves: list[Move] = []
        for k, labels in found:
            if k == SET_IMPLEMENTATION:
                node = cast(nodes.LibraryNode, rows[labels[0]][0])
                moves += [
                    Move(kind=k, labels=labels, implementation=name, current=node.implementation, **scoped)
                    for name in implementations(node, self.targets.gpu)
                ]
            elif k in SCOPE_KINDS:
                if self.scope_refusal(k, scope) is None:
                    assert scope is not None, "scope_refusal requires one"
                    entry = cast(nodes.MapEntry, rows[labels[0]][0])
                    held = self.gpu_choices.get(scope, {}).get(labels[0], GpuChoice())
                    current = schedule_text(entry) if k == SET_SCHEDULE_CPU else held.text()
                    moves.append(Move(kind=k, labels=labels, current=current, **scoped))
            else:
                sizes = probe_sizes(k, [rows[label] for label in labels]) if k in SIZED or k == AUTO_VECTORIZE else ()
                moves.append(Move(kind=k, labels=labels, tile_sizes=sizes, **scoped))
        if kind in (None, RELAYOUT) and scope is None and self.relayout_refusal() is None:
            listed = listed_operation(self.unlaid if self.unlaid is not None else self.sdfg)
            if listed is not None:
                moves.append(Move(kind=RELAYOUT, labels=(), operation=listed))
        if kind in (None, ROLLBACK) and not external_calls(self.sdfg):  # moves close at the first kernel
            moves.append(Move(kind=ROLLBACK, labels=()))
        if scope is not None:  # what staging_refusal would refuse is not offered
            moves = [
                move
                for move in moves
                if move.kind not in DEVICE_KINDS or self.staging_refusal(move.kind, scope) is None
            ]
        return [move for move in moves if switched_off(move.kind, self.disabled_families) is None]

    def scope_refusal(self, kind: str, scope: str | None) -> str | None:
        """Why ``kind`` takes no row of ``scope`` (the program when ``None``), or ``None``: a schedule belongs to the
        device of the kernel it runs in."""
        if kind not in SCOPE_KINDS:
            return None
        if scope is None:
            return Refused(
                f"{kind} sets the schedule of a doall inside a kernel: name its scope (list_kernels).", Refusal.ARGUMENT
            )
        device = kernel_device(self.kernel(scope))
        if kind == SET_SCHEDULE_CPU and device != "cpu":
            return Refused(f"{scope} runs on the GPU; set-schedule-cpu sets OpenMP schedules.", Refusal.ARGUMENT)
        if kind == SET_SCHEDULE_GPU and device != "gpu":
            return Refused(
                f"{scope} runs on the CPU; set-schedule-gpu takes a kernel placed on the GPU (place).", Refusal.ARGUMENT
            )
        return None

    def keep_valid_gpu_choices(self, scope: str) -> None:
        """Drop the GPU choices of ``scope`` that its body, after a move, no longer takes: a doall that moved out of or
        into the kernel doall, or one whose extent changed. A dropped choice runs as DaCe's default."""
        held = self.gpu_choices.get(scope)
        if not held:
            return
        rows = tree_rows(self.move_programs(scope)[0])
        kept: dict[str, GpuChoice] = {}
        for label, choice in held.items():  # in the order they were made, each checked against those kept before it
            if label not in rows:
                continue
            again = plan_set_schedule_gpu(rows[label], choice.block, None if choice.block else choice.schedule, kept)
            if again == choice:
                kept[label] = choice
        self.gpu_choices = {**self.gpu_choices, scope: kept}

    def staging_refusal(self, kind: str, scope: str) -> str | None:
        """Why ``kind`` would undo or clash with what the GPU kernel ``scope`` holds, or ``None``. A placement drops
        ``stage-loads-shared``, and staging adds a thread-block map and a block of its own."""
        # TODO(owner): a device move that survives a new placement would lift both refusals (schedule-moves.md).
        staged = [
            ext.name
            for ext in external_calls(self.sdfg)
            if ext.standalone_sdfg is not None
            and any(desc.storage == dace.StorageType.GPU_Shared for desc in ext.standalone_sdfg.arrays.values())
        ]
        if kind == SET_SCHEDULE_GPU and staged:
            return Refused(
                f"{', '.join(staged)} staged loads through shared memory, which placing {scope} again drops; set the GPU "
                "schedules first and stage after, or roll the staging back.",
                Refusal.ARGUMENT,
            )
        if kind in DEVICE_KINDS and any(
            choice.block or choice.schedule != "sequential" for choice in self.gpu_choices.get(scope, {}).values()
        ):
            return Refused(
                f"{scope} has a GPU schedule, and staging sets a thread block of its own; take one or the other.",
                Refusal.ARGUMENT,
            )
        return None

    def set_gpu_schedule(
        self, names: tuple[str, ...], scope: str, block: Sequence[int], schedule: str | None
    ) -> MoveResult:
        """``set-schedule-gpu`` on the doall ``names`` of the GPU kernel ``scope``: the choice is kept per kernel and
        set on its placed body, now and after every later placement, which resets thread blocks and schedules."""
        kind = SET_SCHEDULE_GPU
        structured, _ = self.move_programs(scope)
        rows = self.named_rows(kind, names, tree_rows(structured))
        if isinstance(rows, MoveResult):
            return rows
        held = self.gpu_choices.get(scope, {})
        choice = plan_set_schedule_gpu(rows[0], block, schedule, held)
        if not isinstance(choice, GpuChoice):
            return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=choice)
        with self.atomic():
            self.gpu_choices = {**self.gpu_choices, scope: {**held, names[0]: choice}}
            self.lower_again()
            self.bump()
            done = MoveResult(
                status=MoveStatus.APPLIED, kind=kind, labels=names, reason=f"GPU schedule {choice.text()}", scope=scope
            )
            self.applied.append(done)
            self.remember()
        return done

    def relayout_refusal(self) -> str | None:
        """Why no argument can be laid out now, or None: the layouts come first, before every other change."""
        others = sorted({move.kind for move in self.applied if move.kind != RELAYOUT})
        if others:
            return Refused(
                f"lay the arguments out before any other change; {others} came first (roll them back, or keep the "
                "layout as it is).",
                Refusal.ARGUMENT,
            )
        return None

    def relayout(self, operation: Mapping[str, object]) -> MoveResult:
        """Lay one argument out by ``operation`` (one relayout operation, as the bench's ``requests`` take it): the
        program before its first relayout is laid out again with every operation so far and this one, so each names
        the digits the ones before it left, as the harness reads the list."""
        refused = self.relayout_refusal()
        if refused is not None:
            return MoveResult(status=MoveStatus.ILLEGAL, kind=RELAYOUT, labels=(), reason=refused)
        unlaid = self.unlaid if self.unlaid is not None else self.sdfg
        try:
            op = layout_operation(operation, unlaid)
            laid = laid_out(unlaid, (*self.layout, op))
        except ValueError as err:  # relayout's refusal, naming what it cannot lay out
            reason = Refused(str(err), Refusal.ARGUMENT)
            return MoveResult(status=MoveStatus.ILLEGAL, kind=RELAYOUT, labels=(), reason=reason)
        with self.atomic():
            self.unlaid, self.layout, self.sdfg = unlaid, (*self.layout, op), laid
            self.unplaced = None
            self.bump()
            self.sdfg.validate()
            reason = f"relayout {op.action.value}"
            done = MoveResult(status=MoveStatus.APPLIED, kind=RELAYOUT, labels=(), reason=reason)
            self.applied.append(done)
            self.remember()
        return done

    def apply_move(
        self,
        kind: str,
        labels: Sequence[str],
        tile_sizes: Sequence[int] = (),
        implementation: str | None = None,
        max_iterations: int = PROBE_UNROLL,
        config_values: Mapping[str, int | float] | None = None,
        scope: str | None = None,
        operation: Mapping[str, object] | None = None,
        isa: str = DEFAULT_ISA,
        force: bool = False,
        schedule: str | None = None,
        chunk: int | None = None,
        threads: int | None = None,
    ) -> MoveResult:
        """Apply one fusion, fission or interchange to the tree rows ``labels`` name; their labels retire, and the rows
        the move makes are named afresh. Inside a kernel (``scope``), a structured move rewrites its body before
        placement, which is placed again; a device move (``stage-loads-shared``) rewrites the body as placed, and a
        later placement drops it.

        :param kind: A key of :data:`~nestforge.stages.moves.MOVE_SHAPES`.
        :param labels: Tree labels in the order the kind takes them: first then second for a fusion, outer then inner
            for an interchange, the map then the body block to cut after for ``subgraph-fission``.
        :param tile_sizes: For ``tile-doall``, one positive size per dimension of the kernel; for ``tile-for``, one; for
            ``stage-loads-shared``, the thread block, one per dimension of the kernel (at most three).
        :param implementation: For ``set-library-implementation``, one :meth:`list_moves` offers for the row.
        :param max_iterations: For ``unroll-for`` and ``unroll-doall``, the largest trip count to unroll.
        :param config_values: For ``propagate-config``, the run-time values of scalar arguments and free symbols;
            the program becomes a dispatch between the program specialized for them and the program as it was.
        :param scope: The kernel whose body holds the rows, as :meth:`list_kernels` names it; ``None``: the program.
        :param operation: For ``relayout``, one relayout operation on an argument (:meth:`relayout`).
        :param isa: For ``auto-vectorize-scope``, the ISA the tiles are for (``tile_sizes``: their lanes).
        :param force: For ``for-to-doall``, apply past a refused dependence check: the caller vouches for the arrays.
        :param schedule: For ``set-schedule-cpu``, the OpenMP schedule; ``chunk`` and ``threads`` go with it.
        :returns: The outcome; nothing but ``applied`` touches the program, and a move that raises leaves it as it was.
        :raises ValueError: ``kind`` is unknown or ``labels`` has the wrong count.
        """
        names = tuple(labels)
        shape = MOVE_SHAPES[check_kind(kind)]
        if len(names) != len(shape):
            raise ValueError(f"{kind} takes {len(shape)} label(s); got {len(names)}")
        family = switched_off(kind, self.disabled_families)
        if family is not None:
            reason = Refused(
                f"{kind} belongs to the family {family}, which this run has switched off.", Refusal.FAMILY_OFF
            )
            return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=reason)
        if kind == ROLLBACK:
            return self.undo()
        if kind == RELAYOUT:
            if scope is not None:
                reason = Refused(
                    "relayout lays out the program's arguments, not a kernel's body: name no scope.", Refusal.ARGUMENT
                )
                return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=reason)
            return self.relayout(operation or {})
        if scope is not None and kind == PROPAGATE_CONFIG:
            reason = Refused(
                "propagate-config specializes the whole program, not a kernel's body: name no scope.", Refusal.ARGUMENT
            )
            return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=reason)
        refusal = self.scope_refusal(kind, scope)
        if refusal is not None:
            return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=refusal)
        if scope is not None and (kind == SET_SCHEDULE_GPU or kind in DEVICE_KINDS):
            staging = self.staging_refusal(kind, scope)
            if staging is not None:
                return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=staging)
        if kind == SET_SCHEDULE_GPU:
            assert scope is not None, "scope_refusal requires one"
            return self.set_gpu_schedule(tuple(names), scope, tile_sizes, schedule)
        structured, device = self.move_programs(scope)
        target = device if kind in DEVICE_KINDS else structured
        rows = self.named_rows(kind, names, None if scope is None else tree_rows(target))
        if isinstance(rows, MoveResult):
            return rows
        config = (config_values, target, self.work_dir / "config")
        plan = plan_move(
            kind,
            rows,
            tile_sizes,
            implementation,
            max_iterations,
            *config,
            isa=isa,
            target=self.targets.canon_target,
            force=force,
            schedule=schedule,
            chunk=chunk,
            threads=threads,
        )
        if kind == SET_IMPLEMENTATION and isinstance(plan, Rewrite):
            node = cast(nodes.LibraryNode, rows[0][0])
            offered = implementations(node, self.targets.gpu)
            if implementation not in offered:
                plan = Refused(
                    f"{names[0]} takes one of {list(offered)} on this target; not {implementation!r}.", Refusal.ARGUMENT
                )
        if not isinstance(plan, Rewrite):
            return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=plan)
        with self.atomic():
            made = plan.commit()
            if kind == EXPAND_LIBRARY:
                # the expansion's maps join the program in canonical form, as rows the other moves take
                settle(cast(list[nodes.NestedSDFG], made), self.targets)
            if kind == PROPAGATE_CONFIG:
                # a new program whose branches call nests; the inlining below makes their blocks rows
                target = cast(dace.SDFG, made)
                self.sdfg = target
            if kind not in DEVICE_KINDS:
                # a move may leave a nest at the top of a state (MapFission does), which only canonical form keeps
                # correct
                inline_top_level_nsdfgs(target)
                structural_cleanup(target)
                # a copy the move made, or one the cleanup left, is a lib_copy row
                explicit_copies(target)
                # an interchange that builds a nest leaves its outer memlets whole (#24); fusions and the schedule
                # inference read them
                narrow_nested_memlets(target)
            # setting an implementation changes a property of the row, not the row: its label stays
            consumed = () if kind == SET_IMPLEMENTATION else names
            if scope is None:
                self.unplaced = None
            else:
                self.namer.retire(consumed)
                normalize_labels(target, self.namer)
                consumed = ()
                if self.unplaced is not None and kind not in DEVICE_KINDS:
                    self.keep_valid_gpu_choices(scope)
                    self.lower_again()
            self.bump(consumed)
            # a move DaCe's own check let through may still build what its validation refuses: refused here, undone
            target.validate()
            forced = plan.name == FORCED_LOOP_TO_MAP
            done = MoveResult(
                status=MoveStatus.APPLIED,
                kind=kind,
                labels=names,
                reason=plan.name,
                scope=scope,
                forced=forced,
                warning=FORCED_WARNING if forced else None,
            )
            self.applied.append(done)
            self.remember()
        return done

    def metrics(self, label: str) -> str:
        """Symbolic work and depth of a top-level map, a top-level loop or a kernel, by DaCe's analyses."""
        row = self.row_index().get(label)
        if row is None:
            return f"no tree row is labeled {label}."
        obj = row[0]
        if isinstance(obj, ExternalCall):
            return f"{label}: {sdfg_metrics(node_boundary(obj).standalone_sdfg).suffix()}"
        if not isinstance(obj, (nodes.MapEntry, LoopRegion)):
            return f"{label} is a {type(obj).__name__}; metrics cover top-level maps, top-level loops and kernels."
        try:
            return f"{label}: {scope_metrics(self.sdfg, obj).suffix()}"
        except TypeError as err:  # scope_metrics' own refusal: not a top-level map or loop
            return str(err)
        except WORK_DEPTH_FAILURES as err:
            return f"{label}: no metrics, work-depth analysis failed: {type(err).__name__}: {err}"

    @undone_on_failure
    def default_moves(self) -> str:
        """Stage 2 default: the wavefronts canonicalization left to the moves, then canonicalization's fusion stage and
        the stages after it, each unless its family is switched off."""
        if not self.targets.wavefront and WAVEFRONT_FAMILY not in self.disabled_families:
            skew_wavefronts(self.sdfg, self.targets)
        if DEFAULT_MOVES_FAMILY in self.disabled_families:
            return self.finish_moves()
        fuse_and_finish(self.sdfg, self.targets)
        self.unplaced = None
        self.bump()
        return self.describe()

    @undone_on_failure
    def finish_moves(self) -> str:
        """The stages after fusion, once :meth:`apply_move` chose the granularity by hand."""
        finish(self.sdfg, self.targets)
        narrow_nested_memlets(self.sdfg)
        self.unplaced = None
        self.bump()
        return self.describe()

    # Stage 3: scopes

    def end_moves(self) -> None:
        """The copies the moves took as ``lib_copy`` rows become copy edges again: a scope makes no kernel of one."""
        if implicit_copies(self.sdfg):
            self.bump()

    def keep_unscoped(self) -> None:
        """Keep the program for :meth:`rewind` while it has no kernel."""
        if not external_calls(self.sdfg):
            self.unscoped = copy.deepcopy(self.sdfg), copy.deepcopy(self.namer), list(self.applied)

    def rewind(self) -> str:
        """Back to the program before its first kernel, row labels and :attr:`applied` included, so the moves work on
        it again; the scopes, the placement and every kernel's code are dropped. Without a kept program this changes
        nothing."""
        if self.unscoped is not None:
            # a copy: a failure puts the kept program back through :meth:`atomic`
            self.sdfg, self.namer, self.applied = copy.deepcopy(self.unscoped)
            self.unscoped, self.unplaced, self.focus = None, None, None
            # the kernels are gone, and their names and labels are handed out again
            self.gpu_choices = {}
            self.bump()
        return self.describe()

    @undone_on_failure
    def default_scopes(self) -> list[KernelScope]:
        """Stage 3 default (:func:`~nestforge.stages.scopes.default_units`): parallel maps, loops and conditionals of
        the program by its rules, large maps split first; returns the new kernels."""
        self.end_moves()
        lowered = lower_default_scopes(self.sdfg, self.targets)
        if lowered:
            self.unplaced = None
            self.bump()
        return [
            KernelScope(
                name=ext.name,
                reads=tuple(boundary.inputs),
                writes=tuple(boundary.outputs),
                symbols=tuple(boundary.symbols),
                parallel=is_parallel_kernel(ext),
            )
            for ext, boundary in lowered
        ]

    @undone_on_failure
    def persist_temporaries(self, min_bytes: int = PERSISTENT_MIN_BYTES) -> MoveResult:
        """Option, any time after the moves: allocate each large temporary of the program once, at init, instead of on
        every call (:func:`~nestforge.build.sdfg.make_persistent`). It changes no loop; it saves the page faults of
        fresh memory on each call. Needs :attr:`sizes`, to know which temporaries are large.

        :param min_bytes: Smallest temporary, in bytes at :attr:`sizes`, that is kept.
        :returns: ``applied`` with the names as ``reason``, or why nothing changed.
        """
        if not self.sizes:
            reason = Refused(
                "the sizes are not set, so no temporary's size is known; set Session.sizes first.", Refusal.ARGUMENT
            )
            return MoveResult(status=MoveStatus.ILLEGAL, kind="persist-temporaries", labels=(), reason=reason)
        names = make_persistent(self.sdfg, self.sizes, min_bytes)
        if not names:
            reason = Refused(
                f"no temporary of the program is a heap array of {min_bytes} bytes or more at these sizes.",
                Refusal.WRONG_ROW_KIND,
            )
            return MoveResult(status=MoveStatus.ILLEGAL, kind="persist-temporaries", labels=(), reason=reason)
        if self.unplaced is not None:
            make_persistent(self.unplaced, self.sizes, min_bytes)
        self.bump()
        done = MoveResult(status=MoveStatus.APPLIED, kind="persist-temporaries", labels=(), reason=", ".join(names))
        self.applied.append(done)
        return done

    @undone_on_failure
    def set_scope(self, labels: Sequence[str]) -> MoveResult:
        """Make one kernel of the regions ``labels`` name: several top-level maps of one state, or a straight run of
        blocks of one control-flow region, at any depth. The kernel written in stage 5 fuses what no move could.

        :param labels: Tree labels of the maps, or of the blocks, as :meth:`describe` printed them.
        :returns: ``applied`` with the new kernel's name as ``reason``, or why nothing changed.
        """
        names = tuple(labels)
        if self.unplaced is not None:
            # scoping the lowered program would drop the program placement starts from
            reason = Refused(
                "the kernels are placed: call unplace first, which keeps the scopes and drops the placement.",
                Refusal.ARGUMENT,
            )
            return MoveResult(status=MoveStatus.ILLEGAL, kind="define-scope", labels=names, reason=reason)
        self.end_moves()
        rows = self.named_rows("define-scope", names)
        if isinstance(rows, MoveResult):
            return rows
        unscoped = self.unscoped
        self.keep_unscoped()
        lowered = lower_group_to_external_call(self.sdfg, rows)
        if isinstance(lowered, str):
            self.unscoped = unscoped  # a refused scope is not journaled: it keeps nothing
            reason = lowered if isinstance(lowered, Refused) else Refused(lowered, Refusal.WRONG_ROW_KIND)
            return MoveResult(status=MoveStatus.ILLEGAL, kind="define-scope", labels=names, reason=reason)
        self.unplaced = None
        self.bump(names)
        done = MoveResult(status=MoveStatus.APPLIED, kind="define-scope", labels=names, reason=lowered[0].name)
        self.applied.append(done)
        return done

    @undone_on_failure
    def unset_scope(self, name: str) -> MoveResult:
        """Inline the kernel ``name`` back into the program: the inverse of :meth:`set_scope`. A placement is
        dropped first, as a new :meth:`place` would; its kernel source and tuning are dropped with the kernel.

        :returns: ``applied`` with ``name`` as ``reason``, or why nothing changed.
        """
        calls = {ext.name: ext for ext in external_calls(self.sdfg)}
        if name not in calls:
            reason = Refused(
                f"no kernel is named {name}; kernels are {', '.join(sorted(calls))}.", Refusal.UNKNOWN_LABEL
            )
            return MoveResult(status=MoveStatus.NOT_FOUND, kind="unset-scope", labels=(name,), reason=reason)
        if calls[name].standalone_sdfg is None:
            reason = Refused(f"{name} keeps no SDFG of its nest to inline.", Refusal.WRONG_ROW_KIND)
            return MoveResult(status=MoveStatus.ILLEGAL, kind="unset-scope", labels=(name,), reason=reason)
        if self.unplaced is not None:
            self.sdfg, self.unplaced = copy.deepcopy(self.unplaced), None
        ext = self.kernel(name)
        assert ext.standalone_sdfg is not None, "placement keeps every kernel's nest"
        # a container the kernel keeps to itself (no connector) is an argument of its standalone SDFG only
        connected = {data_name(conn) for conn in (*ext.in_connectors, *ext.out_connectors)}
        for array, desc in ext.standalone_sdfg.arrays.items():
            if array not in connected:
                desc.transient = True
        ext.expand(find_state_of_node(self.sdfg, ext), "DaceReference")
        inline_top_level_nsdfgs(self.sdfg)
        # its tuning goes with the kernel: a later kernel may take its name
        self.gpu_choices = {kernel: held for kernel, held in self.gpu_choices.items() if kernel != name}
        self.bump()
        done = MoveResult(status=MoveStatus.APPLIED, kind="unset-scope", labels=(name,), reason=name)
        self.applied.append(done)
        return done

    def kernel_graph(self) -> KernelGraph:
        """What can reach every kernel argument and program output (:func:`kernel_dependencies`), once per change."""
        if self.kernel_deps is None:
            self.kernel_deps = kernel_dependencies(self.sdfg)
        return self.kernel_deps

    def kernel(self, name: str) -> ExternalCall:
        """The kernel node named ``name``."""
        calls = {ext.name: ext for ext in external_calls(self.sdfg)}
        if name not in calls:
            raise KeyError(f"no kernel is named {name!r}; kernels are {sorted(calls)}")
        return calls[name]

    def list_kernels(self) -> list[KernelInfo]:
        """Every kernel, in program order."""
        graph = self.kernel_graph()
        out: list[KernelInfo] = []
        for name in graph.kernels:
            ext, arguments = self.kernel(name), graph.arguments(name)
            out.append(
                KernelInfo(
                    name=name,
                    device=kernel_device(ext),
                    parallel=is_parallel_kernel(ext),
                    inputs=tuple(edge.arg for edge in arguments if edge.role == "input"),
                    outputs=tuple(sorted(data_name(conn) for conn in ext.out_connectors)),
                    symbols=tuple(edge.arg for edge in arguments if edge.role == "symbol"),
                    depends={edge.arg: edge.labels() for edge in arguments},
                    carried={edge.arg: list(edge.carried) for edge in arguments if edge.carried},
                )
            )
        return out

    # Stage 4: placement

    @undone_on_failure
    def place(self, devices: dict[str, str] | None = None) -> PlaceResult:
        """Run each kernel on a device. Kernels ``devices`` leaves out keep the default: parallel kernels on the GPU
        when it is a target, the rest on the CPU. A GPU kernel is offloaded inside its call, copying its arguments at
        its own boundary; :meth:`insert_copies` moves those copies into the program. A later call starts again from
        the program before the first placement.

        :param devices: Kernel name -> ``cpu`` or ``gpu``.
        """
        if self.unplaced is None:
            self.end_moves()
        chosen = default_devices(self.sdfg, self.targets) | (devices or {})
        refusal = check_devices(self.sdfg, self.targets, chosen)
        if refusal is not None:
            return PlaceResult(status=MoveStatus.ILLEGAL, reason=refusal)
        interfaces, builds = kernel_interfaces(self.sdfg), self.builds
        if self.unplaced is not None:
            self.sdfg = copy.deepcopy(self.unplaced)
            self.kernel_deps = None
        moves = self.transfers(chosen)
        self.unplaced = copy.deepcopy(self.sdfg)
        return self.placed(place(self.sdfg, chosen), moves, interfaces, builds)

    @undone_on_failure
    def insert_copies(self) -> PlaceResult:
        """DaCe's offloading of the placed program: the GPU kernels' operands move to device memory and the host/device
        copies sit at the kernels' boundaries in the program, so adjacent GPU kernels share them."""
        if self.unplaced is None:
            return PlaceResult(status=MoveStatus.ILLEGAL, reason="no kernel is placed yet; call place first.")
        devices = self.devices()
        moves = self.transfers(devices)
        interfaces, builds = kernel_interfaces(self.sdfg), self.builds
        self.sdfg = copy.deepcopy(self.unplaced)
        return self.placed(insert_copies(self.sdfg, devices), moves, interfaces, builds)

    @undone_on_failure
    def unplace(self) -> MoveResult:
        """Back to the program before placement, its kernels kept: the inverse of :meth:`place` and
        :meth:`insert_copies`, so :meth:`set_scope` and :meth:`unset_scope` apply again. Structured moves inside a
        kernel stay, as they rewrote its body before placement; a device move (``stage-loads-shared``) and the
        kernels' code go with the placement.

        :returns: ``applied``, or why nothing changed.
        """
        if self.unplaced is None:
            reason = Refused("no kernel is placed; there is no placement to undo.", Refusal.ARGUMENT)
            return MoveResult(status=MoveStatus.ILLEGAL, kind="unplace", labels=(), reason=reason)
        # a copy: a failure puts the kept program back through :meth:`atomic`
        self.sdfg, self.unplaced = copy.deepcopy(self.unplaced), None
        self.bump()
        done = MoveResult(status=MoveStatus.APPLIED, kind="unplace", labels=(), reason=", ".join(self.devices()))
        self.applied.append(done)
        return done

    def placed(
        self,
        placement: Placement,
        moves: tuple[Transfer, ...] | None,
        interfaces: dict[str, tuple[object, ...]],
        builds: dict[str, KernelBuild],
    ) -> PlaceResult:
        """Bump the program after a placement; a build stays linked while its kernel's ``interfaces`` entry holds."""
        set_gpu_choices(self.sdfg, self.gpu_choices)
        self.bump()
        # a build stays linked while placement leaves its kernel's device and argument storage as they were
        placed = kernel_interfaces(self.sdfg)
        for name, build in builds.items():
            if placed.get(name, ()) == interfaces.get(name):
                self.bind(name, build)
        return PlaceResult(
            status=MoveStatus.APPLIED, devices=dict(placement.devices), copies=tuple(placement.copies), transfers=moves
        )

    def lower_again(self) -> None:
        """The placed program again from :attr:`unplaced`, on the same devices and with copies where they were."""
        devices, lower = self.devices(), insert_copies if has_device_memory(self.sdfg) else place
        self.sdfg = copy.deepcopy(self.unplaced)
        lower(self.sdfg, devices)
        set_gpu_choices(self.sdfg, self.gpu_choices)

    def devices(self) -> dict[str, str]:
        """Every kernel's device."""
        return {ext.name: kernel_device(ext) for ext in external_calls(self.sdfg)}

    def transfers(self, devices: dict[str, str]) -> tuple[Transfer, ...] | None:
        """The transfers the kernel DAG implies for ``devices``, or ``None`` when its analysis refuses the program."""
        try:
            return tuple(transfers(self.kernel_graph(), devices))
        except UnsupportedProgram:
            return None

    def default_placement(self) -> PlaceResult | None:
        """Stage 4 default: :meth:`place` unless placed, then :meth:`insert_copies` while a GPU kernel copies inside
        its call; ``None`` when both are done."""
        done = self.place() if self.unplaced is None else None
        if "gpu" in self.devices().values() and not has_device_memory(self.sdfg):
            done = self.insert_copies()
        return done

    # Stage 5: kernels

    def default_kernel(self, name: str) -> KernelSource:
        """The kernel's CPF unit for its device, rendered once per change of the program."""
        if name not in self.sources:
            ext = self.kernel(name)
            self.sources[name] = schedule_kernel(ext, node_boundary(ext), self.work_dir / name / "kernel")
        return self.sources[name]

    def first_variant(self, device: str) -> Variant:
        variants = device_variants(device)
        if not variants:
            raise LookupError(f"no {device} toolchain on this machine")
        return variants[0]

    def bind(self, name: str, build: KernelBuild) -> None:
        """Link ``build`` into the kernel's ``ExternalCall``."""
        src = build.source
        runtime = kernel_runtime_libraries(src, build.variant.compiler)
        ext = self.kernel(name)
        if kernel_device(ext) == "gpu" and not has_device_memory(self.sdfg):
            raise ValueError(f"{name} takes device memory, which the program passes once insert_copies ran.")
        use_kernel_library(ext, build.archive, src, runtime)
        self.builds[name] = build
        self.measured = None

    def optimize_kernel(self, name: str) -> KernelLibrary:
        """Stage 5 default: render the kernel's CPF unit, build it with the first configuration stage 6 sweeps for
        its device, and link it into the kernel's ``ExternalCall``."""
        src = self.default_kernel(name)
        variant = self.first_variant(src.device)
        with self.stopper.armed():
            archive = build_kernel_library(src, variant.compiler, list(variant.flags), self.work_dir / name / "library")
        self.bind(name, KernelBuild(src, variant, archive))
        return KernelLibrary(
            kernel=name,
            symbol=src.symbol,
            abi_order=tuple(src.abi_order),
            unit=src.unit,
            library=archive,
            variant=variant.label,
        )

    def focus_kernel(self, name: str) -> str:
        """Scope the kernel work to the kernel ``name``; returns it."""
        self.focus = self.kernel(name).name
        return self.focus

    def kernel_source(self, name: str) -> str:
        """The source of the kernel's current library, or of its default CPF unit before one is built."""
        build = self.builds.get(name)
        return kernel_text(build.source if build is not None else self.default_kernel(name))

    def set_kernel_source(
        self, name: str, source: str, language: str, reps: int = TIMED_REPS, checked: bool = True
    ) -> KernelOutcome:
        """Build a kernel source with the kernel's entry, validate it against the kernel's Python oracle, and link it
        when it matches.

        :param source: A translation unit defining ``extern "C" void <name>(...)`` with the parameters, in order, of
            the default unit (:meth:`kernel_source` before any change). A GPU kernel may be two units, host then
            device, split by the line ``DEVICE_UNIT_MARKER``, as :meth:`kernel_source` returns it.
        :param language: ``cpp`` (C++ with OpenMP) for a CPU kernel; for a GPU kernel ``cuda`` or ``hip``, the
            dialect of this machine's GPU backend.
        :param checked: False for a rebuild's repeat: the source matched its oracle when it was written, so it is
            built and linked unchecked.
        :returns: The outcome; ``ok`` and ``unchecked`` change the kernel's library. ``unchecked`` had no oracle to match
            (:func:`~nestforge.stages.kernel.measure_kernel`), so the whole program checks the kernel.
        """
        default = self.default_kernel(name)
        if language != kernel_language(default.device):
            return KernelOutcome(
                kernel=name,
                status=KernelStatus.REFUSED,
                reason=f"a {default.device} kernel is {kernel_language(default.device)}.",
            )
        try:
            params = [p.name for p in parse_params(raw_signature(source, name))]
        except (LookupError, ValueError) as err:
            return KernelOutcome(kernel=name, status=KernelStatus.REFUSED, reason=str(err))
        if params != default.abi_order:
            return KernelOutcome(
                kernel=name,
                status=KernelStatus.REFUSED,
                reason=f"the entry takes {params}; it must take {default.abi_order}.",
            )
        attempt = self.work_dir / name / f"agent{len(list((self.work_dir / name).glob('agent*')))}"
        attempt.mkdir(parents=True)
        host, device_text = split_kernel_text(source)
        suffix = kernel_form(default.device).suffix
        unit = attempt / f"{name}{suffix}"
        unit.write_text(host)
        device_unit = None
        if device_text is not None:
            device_unit = attempt / f"{name}{DEVICE_UNIT_SUFFIX}{suffix}"
            device_unit.write_text(device_text)
        src = KernelSource(
            name, unit, list(default.abi_order), default.boundary, default.device, device_unit, list(default.link)
        )
        variant = self.first_variant(src.device)
        with self.stopper.armed():
            try:
                archive = build_kernel_library(src, variant.compiler, list(variant.flags), attempt)
            except RuntimeError as err:
                return KernelOutcome(kernel=name, status=KernelStatus.BUILD_FAILED, reason=str(err)[-ERROR_TAIL:])
            verdict = (
                self.kernel_verdict(name, src, archive, reps, variant.fp_mode)
                if checked
                else unchecked_verdict(variant.fp_mode, REPEATED)
            )
        if not verdict.ok and not verdict.unchecked:
            reason = verdict.error or f"max rel err {verdict.md_rel:.3g} at {verdict.fp_mode}"
            return KernelOutcome(kernel=name, status=KernelStatus.WRONG, reason=reason)
        self.bind(name, KernelBuild(src, variant, archive, verdict))
        self.applied.append(MoveResult(status=MoveStatus.APPLIED, kind="write-kernel", labels=(name,), reason=""))
        if verdict.unchecked:
            return KernelOutcome(kernel=name, status=KernelStatus.UNCHECKED, reason=verdict.unchecked)
        return KernelOutcome(kernel=name, status=KernelStatus.OK, time_us=verdict.time_us)

    def kernel_verdict(self, name: str, src: KernelSource, archive: Path, reps: int, fp_mode: str) -> KernelVerdict:
        """``archive`` against the kernel's oracle, unchecked when the oracle has no value for a symbol it takes."""
        computed = computed_symbols(src, self.kernel_sizes(name))
        if computed:
            return unchecked_verdict(fp_mode, computed_symbols_reason(computed))
        unspelled = self.prepare_kernel(name).unspelled
        if unspelled:
            return unchecked_verdict(fp_mode, f"{unspelled}; the whole program checks it")
        return measure_kernel(archive, src, self.kernel_probe(name, src), reps, fp_mode)

    def prepare_kernel(self, name: str) -> Prepared:
        """The kernel's NumPy oracle, from the boundary its source was rendered from. Not the node's: binding a library
        sets the node's ``abi_order`` to the built entry's parameters, which drop a symbol the body never reads
        (``NSUB`` of a nest inside the substep loop), and the oracle would no longer take what validation passes."""
        if name not in self.prepared:
            self.prepared[name] = prepare(self.default_kernel(name).boundary, name)
        return self.prepared[name]

    def kernel_probe(self, name: str, src: KernelSource) -> Probe:
        """What builds of the kernel are measured on; without :attr:`check_sizes` the oracle runs at the kernel's sizes
        however long it takes."""
        prep, sizes = self.prepare_kernel(name), self.kernel_sizes(name)
        if self.check_sizes is None:
            return full_probe(prep, src.boundary, sizes)
        check = {**self.loop_values(name, self.check_sizes), **self.check_sizes}
        full = prep.numpy_source not in self.oracles_out_of_time
        probe = bounded_probe(prep, src.boundary, sizes, check, full)
        if probe.checked is not probe.timed:
            self.oracles_out_of_time.add(prep.numpy_source)
        return probe

    def need_sizes(self) -> dict[str, SymbolValue]:
        if not self.sizes:
            raise ValueError("this session has no sizes; pass Session(..., sizes={symbol: value})")
        return self.sizes

    def kernel_sizes(self, name: str) -> dict[str, SymbolValue]:
        """The sizes plus, for a kernel inside a sequential loop, a middle value of each loop iterator it takes;
        a value given in ``sizes`` wins."""
        sizes = self.need_sizes()
        return {**self.loop_values(name, sizes), **sizes}

    def loop_values(self, name: str, sizes: Mapping[str, SymbolValue]) -> dict[str, int]:
        """A middle value at ``sizes`` of each loop iterator around the kernel."""
        return loop_symbol_values(find_state_of_node(self.sdfg, self.kernel(name)), sizes)

    # Stage 6: variants

    def sweep(
        self, name: str, reps: int = TIMED_REPS, compilers: list[str] | None = None, variant: str | None = None
    ) -> Sweep:
        """Build and time the kernel's current source under every configuration of its device, link the fastest
        correct one, and summarize the sweep; ``config`` is the winner's compiler, FP mode, cost model, flags, time.

        :param compilers: Toolchain names to keep (``gcc``, ``clang``, ``nvcc``, ``nvcc-13.1``, ``hipcc``, ...); every one
            discovered for the kernel's device when ``None``.
        :param variant: For a rebuild's repeat, the winner's label: it is built and linked unchecked, nothing is timed.
        """
        build = self.builds.get(name)
        src = build.source if build is not None else self.default_kernel(name)
        if variant is not None:
            return self.link_variant(name, src, variant, compilers)
        sizes = self.kernel_sizes(name)
        computed = computed_symbols(src, sizes)
        if computed:
            raise ValueError(
                f"{name} takes {computed}, which only the running program computes; its oracle has no value for "
                "them, so it cannot be swept. finalize builds its first configuration."
            )
        unspelled = self.prepare_kernel(name).unspelled
        if unspelled:
            raise ValueError(f"{name}: {unspelled}, so it cannot be swept. finalize builds its first configuration.")
        variants = device_variants(src.device, compilers)
        key = self.sweep_key(name, src)
        cells = self.sweep_cells.setdefault(key, {})
        out_dir = self.swept_dir / f"{name}_cells_{list(self.sweep_cells).index(key)}"
        with self.stopper.armed():
            result = select_variant(src, self.kernel_probe(name, src), reps, variants, out_dir, cells)
        winner = result.winner
        if winner is not None and winner.archive is not None:
            build = KernelBuild(src, winner.variant, winner.archive, winner.verdict)
            # a folder of its own: a count reused by a sweep again of a key would overwrite a linked winner
            self.swept[key] = kept(build, Path(tempfile.mkdtemp(prefix=f"{name}_", dir=self.swept_dir)))
            self.bind(name, self.swept[key])
        if winner is None:
            return Sweep(kernel=name, cells=len(result.cells), winner=None, config=None)
        variant = winner.variant
        config = Config(
            compiler=variant.toolchain,
            fp_mode=variant.fp_mode,
            cost_model=variant.cost_model,
            flags=tuple(variant.flags),
            time_us=winner.verdict.time_us,
        )
        return Sweep(kernel=name, cells=len(result.cells), winner=variant.label, config=config)

    def link_variant(self, name: str, src: KernelSource, label: str, compilers: list[str] | None) -> Sweep:
        """The sweep winner ``label`` of ``src``, built and linked again unchecked: it won, correct, when it was swept."""
        found = [v for v in device_variants(src.device, compilers) if v.label == label]
        if not found:
            raise LookupError(f"{name}: no toolchain of this machine builds the sweep winner {label} again")
        variant = found[0]
        self.swept_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=self.swept_dir))
        with self.stopper.armed():
            archive = build_kernel_library(src, variant.compiler, list(variant.flags), out_dir)
        self.bind(name, KernelBuild(src, variant, archive, unchecked_verdict(variant.fp_mode, REPEATED)))
        return Sweep(kernel=name, cells=0, winner=label, config=None)

    def sweep_key(self, name: str, src: KernelSource) -> SweepKey:
        units = tuple(unit.read_text() for unit in src.units)
        return name, units, tuple(sorted(self.kernel_sizes(name).items()))

    def default_build(self, name: str) -> None:
        """Build and link a kernel that has no build: stage 5's first configuration, or under ``codegen="sweep"``
        the winner of stage 6's sweep (one swept before for the same source and sizes is relinked, not swept again;
        a sweep without a correct winner falls back to the first configuration, and so does a kernel taking a symbol
        only the running program computes: its oracle has no value for it)."""
        if self.codegen == "first":
            self.first_build(name)
            return
        src = self.default_kernel(name)
        if computed_symbols(src, self.kernel_sizes(name)):
            self.optimize_kernel(name)
            return
        known = self.swept.get(self.sweep_key(name, src))
        if known is not None:
            self.bind(name, known)
        elif self.sweep(name).winner is None:
            self.optimize_kernel(name)

    def first_build(self, name: str) -> None:
        """Stage 5's default build, or the one made before for the same kernel body and sizes: a move changes few
        kernels, and rendering is most of a kernel's cost, so the key needs no rendering."""
        ext = self.kernel(name)
        body = (kernel_device(ext), ext.standalone_sdfg.hash_sdfg(), *strings(ext.abi_order))
        key = name, body, tuple(sorted(self.kernel_sizes(name).items()))
        known = self.first_builds.get(key)
        if known is None:
            self.optimize_kernel(name)
            known = kept(self.builds[name], self.swept_dir / f"{name}_first_{len(self.first_builds)}")
            self.first_builds[key] = known
        self.bind(name, known)

    # Stage 7: feedback

    def feedback(self, reps: int = TIMED_REPS) -> str:
        """A few lines of hints for the next round, most important first, from every kernel's current library:
        its validation and time, and the compiler's vectorizer or register remarks."""
        return feedback.report(self.kernel_evidence(reps))

    def kernel_evidence(self, reps: int = TIMED_REPS) -> list[feedback.Evidence]:
        """Every kernel's validation, time and compiler remarks, building the default library where none is bound."""
        evidence: list[feedback.Evidence] = []
        for ext in external_calls(self.sdfg):
            if ext.name not in self.builds:
                self.default_build(ext.name)
            build = self.builds[ext.name]
            if build.verdict is None:
                with self.stopper.armed():
                    build.verdict = self.kernel_verdict(
                        ext.name, build.source, build.archive, reps, build.variant.fp_mode
                    )
            if build.remarks is None:
                variant = build.variant
                build.remarks = feedback.compiler_remarks(build.source, variant.compiler, list(variant.flags))
            evidence.append(feedback.Evidence(ext.name, build.verdict, build.remarks))
        return evidence

    # Assembly

    def finalize(self, out_dir: Path | None = None) -> Final:
        """The program as a bench library submission, built on a :meth:`trial`: score and submit call this, so the
        program stays in its phase. On the copy a program without scopes gets the default scopes, one never placed on
        a GPU target the default placement, and every kernel without a build its default build
        (:meth:`default_build`), then the program is rendered in the bench C ABI and built, each kernel linked from
        its own archive. A sweep's winner is kept in :attr:`swept`, so an unchanged program is not swept again.

        :param out_dir: Where the library is built; ``<work_dir>/final`` when ``None``.
        """
        if self.bench is None:
            raise ValueError("this session was not started from a bench kernel; set Session.bench to finalize")
        refused = layout_refusal(self.layout)
        if refused is not None:
            raise ValueError(refused)
        # all of it on a trial, so a stop anywhere changes nothing and keeps the session and its sweeps
        with self.stopper.armed():
            trial = self.trial()
            try:
                if not external_calls(trial.sdfg):
                    trial.default_scopes()
                if trial.targets.gpu:
                    trial.default_placement()  # a GPU program is assembled as placed, its copies inserted
                for ext in external_calls(trial.sdfg):
                    if ext.name not in trial.builds:
                        trial.default_build(ext.name)
                variants = {name: build.variant.label for name, build in trial.builds.items()}
                final_dir = out_dir or self.work_dir / "final"
                return assemble(trial.sdfg, self.bench, variants, final_dir, requests_of(self.layout))
            finally:
                # the library holds the kernels' archives; a long session would otherwise keep every trial until it exits
                shutil.rmtree(trial.work_dir, ignore_errors=True)

    def trial(self) -> Session:
        """A copy of this session in its own work directory, at the same stage, for defaults that must not stick."""
        # same labels, same namer state: what was read on this session names the same rows of the copy
        copied = Session(
            copy.deepcopy(self.sdfg),
            self.targets,
            self.name,
            sizes=self.sizes,
            codegen=self.codegen,
            namer=copy.deepcopy(self.namer),
            disabled_families=self.disabled_families,
        )
        # shared: a winner the trial sweeps serves this session's finalize too, kept in this session's folder
        copied.swept, copied.swept_dir = self.swept, self.swept_dir
        copied.first_builds, copied.sweep_cells = self.first_builds, self.sweep_cells
        copied.check_sizes, copied.oracles_out_of_time = self.check_sizes, self.oracles_out_of_time
        copied.stopper = self.stopper
        copied.unplaced = copy.deepcopy(self.unplaced)
        copied.unlaid, copied.layout = self.unlaid, self.layout
        # the copied kernels link this session's libraries, so the trial measures them instead of rebuilding defaults
        copied.sources, copied.builds = dict(self.sources), dict(self.builds)
        return copied

    def performance(
        self,
        inputs: dict[str, np.ndarray],
        reference: Reference | None = None,
        reps: int = TIMED_REPS,
        codegen: Codegen | None = None,
    ) -> Performance:
        """Every phase's feedback at once (:mod:`~nestforge.stages.performance`): on a :meth:`trial`, the default
        scopes when none is defined, the default placement when none is chosen and the default kernel builds, then
        the whole program (checked against ``reference`` when given), its offloading and each kernel are timed. Reused
        for the same arguments until the program changes.

        :param codegen: How the kernels without a build of their own are built for this measurement; the session's
            own ``codegen`` when ``None``."""
        codegen = codegen if codegen is not None else self.codegen
        cached = self.measured
        if cached is not None and cached[0] is inputs and cached[1] is reference and cached[2:4] == (reps, codegen):
            return cached[4]
        with self.stopper.armed():
            trial = self.trial()
            trial.codegen = codegen
            try:
                measured = trial.measure(inputs, reference, reps)
            finally:
                # the trial's builds are measured; a long session would otherwise keep every one until it exits
                shutil.rmtree(trial.work_dir, ignore_errors=True)
        self.measured = inputs, reference, reps, codegen, measured
        return measured

    def measure(self, inputs: dict[str, np.ndarray], reference: Reference | None, reps: int) -> Performance:
        """The feedback of :meth:`performance` on this session, a self."""
        if not external_calls(self.sdfg):
            self.default_scopes()
        in_call = copies_in_call(self.sdfg)  # before the default placement inserts their copies
        # as finalize: placing a CPU program changes nothing but drops its kernels' builds
        if self.targets.gpu:
            self.default_placement()
        devices = self.devices()
        moves = self.transfers(devices)
        evidence = self.kernel_evidence(reps)
        if self.layout:
            # the program works on its arguments laid out: they go in laid out and its outputs come back read
            assert self.unlaid is not None, "a laid-out program keeps the program before its layout"
            layouts = argument_layouts(self.unlaid, self.layout)
            sizes = {name: int(value) for name, value in self.sizes.items()}
            inputs = physical_inputs(inputs, layouts, sizes)
            if reference is not None:
                reference = dataclasses.replace(reference, layouts=layouts, sizes=sizes)
        program = time_program(
            self.sdfg, self.work_dir / "program", inputs, reference, self.sizes, reps, "gpu" in devices.values()
        )
        return Performance(
            structured=program,
            offloading=offload_report(self.sdfg, devices, moves, self.sizes, in_call),
            kernels=kernel_reports(evidence, devices),
            hints=feedback.report(evidence),
        )
