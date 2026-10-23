# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the one API over the seven stages, shared by the deterministic driver, humans and agents.

Callers name tree rows by their labels together with the epoch they read them at, and kernels by name. Every
mutation of the program bumps the epoch and regenerates the labels, so a label from before it is refused instead of
acting on a moved graph. Every stage has a deterministic default; the agent calls are the alternatives to it.
"""

from __future__ import annotations

import atexit
import copy
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import ExternalCall, data_name, external_calls
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState

from nestforge.build.arena import TIMED_REPS
from nestforge.build.sdfg import force_register_arrays
from nestforge.build.toolchain import parse_params, raw_signature
from nestforge.corpus.translate import Prepared, prepare
from nestforge.ir.depends import KernelGraph, UnsupportedProgram, kernel_dependencies
from nestforge.ir.extract import find_state_of_node
from nestforge.ir.introspect import (
    WORK_DEPTH_FAILURES,
    Row,
    WorkDepth,
    describe_graph,
    guarded_work_depth,
    tree_rows,
)
from nestforge.ir.loops import loop_symbol_values
from nestforge.ir.names import inline_top_level_nsdfgs, normalize_labels
from nestforge.records import KernelOutcomeStatus, KernelStatus, MoveOutcome, MoveStatus, Record, spelled
from nestforge.stages import feedback
from nestforge.stages.canonicalize import Targets, canonicalize, finish, fuse_and_finish, narrow_nested_memlets
from nestforge.stages.kernel import (
    DEVICE_UNIT_SUFFIX,
    FORMS,
    KernelSource,
    KernelVerdict,
    build_kernel_library,
    kernel_runtime_libraries,
    kernel_text,
    schedule_kernel,
    split_kernel_text,
    unchecked_verdict,
    use_kernel_library,
    validate_kernel,
)
from nestforge.stages.moves import (
    MOVE_SHAPES,
    Rewrite,
    check_kind,
    legal_moves,
    plan_move,
    scope_metrics,
    sdfg_metrics,
)
from nestforge.stages.performance import Performance, Reference, kernel_reports, offload_report, time_program
from nestforge.stages.placement import Transfer, check_devices, default_devices, kernel_device, place, transfers
from nestforge.stages.scopes import (
    is_parallel_kernel,
    lower_group_to_external_call,
    lower_nests_to_external_call,
    node_boundary,
)
from nestforge.stages.variants import Variant, device_variants, select_variant

#: Characters of a failed build's error text a caller gets back.
ERROR_TAIL = 1500

#: The kernel source language each device's kernels are written in.
LANGUAGES = {"cpu": "cpp", "gpu": "cuda"}


class MoveResult(Record):
    """What a labeled call did; ``reason`` names what ran once ``applied``, else why nothing changed."""

    status: MoveOutcome
    kind: str
    labels: tuple[str, ...]
    reason: str


class Move(Record):
    """A legal move, exactly as :meth:`Session.apply_move` takes it back."""

    kind: str
    labels: tuple[str, ...]
    epoch: int


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
    """The library a kernel links now: the unit it was built from, the variant that built it, its last verdict."""

    source: KernelSource
    variant: Variant
    archive: Path
    verdict: KernelVerdict | None = None


def scratch_dir() -> Path:
    """A work directory of the session's own, removed when the process exits."""
    path = Path(tempfile.mkdtemp(prefix="nfsession_"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


class Session:
    """Owner of one program SDFG, driven stage by stage."""

    __slots__ = (
        "sdfg",
        "name",
        "targets",
        "sizes",
        "epoch",
        "rows",
        "expanded",
        "costs",
        "costs_skipped",
        "work_dir",
        "kernel_deps",
        "unplaced",
        "sources",
        "builds",
        "prepared",
        "measured",
    )

    def __init__(
        self,
        sdfg: dace.SDFG,
        targets: Targets | None = None,
        name: str | None = None,
        work_dir: str | None = None,
        sizes: dict[str, int] | None = None,
    ) -> None:
        """:param sizes: Value of every symbol, used to validate and time kernels (stages 5 to 7)."""
        self.sdfg = sdfg
        self.targets = targets if targets is not None else Targets()
        self.name = name or sdfg.label
        self.sizes = dict(sizes) if sizes is not None else {}
        self.epoch = 0
        normalize_labels(sdfg)
        self.rows: dict[str, Row] | None = None
        # labels whose content describe() shows, and the work and depth it prints; both reset each epoch
        self.expanded: set[str] = set()
        self.costs: WorkDepth | None = None
        # why describe() printed no costs this epoch, or None when it printed them
        self.costs_skipped: str | None = None
        self.work_dir = Path(work_dir) if work_dir else scratch_dir()
        self.kernel_deps: KernelGraph | None = None
        # the program before stage 4, so a new placement starts from it
        self.unplaced: dace.SDFG | None = None
        self.sources: dict[str, KernelSource] = {}
        self.builds: dict[str, KernelBuild] = {}
        self.prepared: dict[str, Prepared] = {}
        # :meth:`performance` of the current program, dropped by every change to it
        self.measured: Performance | None = None

    def bump(self) -> None:
        self.epoch += 1
        normalize_labels(self.sdfg)
        # placement decides a kernel's device: pinned to the host stack now, a GPU kernel's arrays would stay there
        force_register_arrays(self.sdfg, kernels=not self.targets.gpu)
        self.rows = None
        self.expanded = set()
        self.costs = None
        self.costs_skipped = None
        self.kernel_deps = None
        self.sources, self.builds, self.prepared = {}, {}, {}
        self.measured = None

    def row_index(self) -> dict[str, Row]:
        """Tree label -> ``(block or node, state)``, built once per epoch."""
        if self.rows is None:
            self.rows = tree_rows(self.sdfg)
        return self.rows

    def named_rows(self, kind: str, names: tuple[str, ...], epoch: int) -> list[Row] | MoveResult:
        """The rows ``names`` label at ``epoch``, or the ``stale`` or ``not-found`` result."""
        if epoch != self.epoch:
            reason = f"labels read at epoch {epoch}; the program is at epoch {self.epoch}. Describe again."
            return MoveResult(status=MoveStatus.STALE, kind=kind, labels=names, reason=reason)
        rows = self.row_index()
        missing = [name for name in names if name not in rows]
        if missing:
            return MoveResult(
                status=MoveStatus.NOT_FOUND,
                kind=kind,
                labels=names,
                reason=f"no tree row is labeled {', '.join(missing)}.",
            )
        return [rows[name] for name in names]

    # Stage 1: canonicalize

    def canonicalize(self) -> str:
        """Canonicalize up to the fusion stage for the session's targets; returns the new tree."""
        canonicalize(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    # Stage 2: moves

    def describe(self, bodies: bool = False, deps: bool = False) -> str:
        """The program as a text tree headed by the epoch; rows are named by the labels the labeled calls take.
        Every loop and state is a row, loop and map rows carry work and depth, and a state shows its kernels only
        once :meth:`expand` opened it. ``deps`` puts each kernel's :meth:`kernel_graph` line under its row."""
        if self.costs is None:
            self.costs, self.costs_skipped = guarded_work_depth(self.sdfg)
        tree = describe_graph(
            self.sdfg,
            bodies=bodies,
            notes=self.deps_line if deps else None,
            epoch=self.epoch,
            expanded=self.expanded,
            costs=self.costs,
        )
        if self.costs_skipped is None:
            return tree
        header, _, rows = tree.partition("\n")
        return f"{header}\n(no work= or depth= this epoch: {self.costs_skipped})\n{rows}"

    def expand(self, labels: Sequence[str], epoch: int) -> str:
        """Show the content of the states and maps ``labels`` name (kernels, nested maps and leaf bodies; a map's
        nested states) until the program changes; returns what opened, or why nothing did."""
        rows = self.named_rows("expand", tuple(labels), epoch)
        if isinstance(rows, MoveResult):
            return f"{spelled(rows.status)}: {rows.reason}"
        closed = [name for name, (obj, _) in zip(labels, rows) if not isinstance(obj, (SDFGState, nodes.MapEntry))]
        if closed:
            return f"illegal: only states and maps expand; {', '.join(closed)} is neither."
        self.expanded.update(labels)
        return f"ok: expanded {', '.join(labels)}"

    def view(self, expand: Sequence[str] = (), collapse: Sequence[str] = (), epoch: int | None = None) -> str:
        """:meth:`collapse` then :meth:`expand` (labels read at ``epoch``), and the tree after them, each note first."""
        notes = [self.collapse(collapse)] if collapse else []
        if expand and epoch is None:
            notes.append("illegal: expand needs the epoch its labels were read at.")
        elif expand:
            notes.append(self.expand(expand, epoch))
        return "\n".join([*notes, self.describe()])

    def run_defaults(self, stage: Literal["moves", "finish", "scopes", "placement"]) -> object:
        """Finish ``stage`` with its default: :meth:`default_moves`, :meth:`finish_moves`, :meth:`default_scopes` or
        :meth:`place`."""
        match stage:
            case "moves":
                return self.default_moves()
            case "finish":
                return self.finish_moves()
            case "scopes":
                return self.default_scopes()
            case "placement":
                return self.place()

    def collapse(self, labels: Sequence[str]) -> str:
        """Hide again what :meth:`expand` opened under ``labels``; a label not open is ignored."""
        self.expanded.difference_update(labels)
        return f"ok: collapsed {', '.join(labels)}"

    def deps_line(self, node: nodes.LibraryNode) -> str | None:
        graph = self.kernel_graph()
        return graph.line(node.label) if node.label in graph.kernels else None

    def list_moves(self, kind: str | None = None) -> list[Move]:
        """Every legal move right now, of ``kind`` or of every kind."""
        return [Move(kind=k, labels=tuple(labels), epoch=self.epoch) for k, labels in legal_moves(self.sdfg, kind)]

    def apply_move(self, kind: str, labels: Sequence[str], epoch: int) -> MoveResult:
        """Apply one fusion, fission or interchange to the tree rows ``labels`` name, as read at ``epoch``.

        :param kind: A key of :data:`~nestforge.stages.moves.MOVE_SHAPES`.
        :param labels: Tree labels in the order the kind takes them: first then second for a fusion, outer then inner
            for an interchange, the map then the body block to cut after for ``subgraph-fission``.
        :param epoch: The epoch :meth:`describe` or :meth:`list_moves` showed with those labels.
        :returns: The outcome; nothing but ``applied`` touches the program.
        :raises ValueError: ``kind`` is unknown or ``labels`` has the wrong count.
        """
        names = tuple(labels)
        shape = MOVE_SHAPES[check_kind(kind)]
        if len(names) != len(shape):
            raise ValueError(f"{kind} takes {len(shape)} label(s); got {len(names)}")
        rows = self.named_rows(kind, names, epoch)
        if isinstance(rows, MoveResult):
            return rows
        plan = plan_move(kind, rows)
        if not isinstance(plan, Rewrite):
            return MoveResult(status=MoveStatus.ILLEGAL, kind=kind, labels=names, reason=plan)
        plan.commit()
        # an interchange that builds a nest leaves its outer memlets whole (#24); fusions and the schedule inference
        # read them
        narrow_nested_memlets(self.sdfg)
        self.unplaced = None
        self.bump()
        return MoveResult(status=MoveStatus.APPLIED, kind=kind, labels=names, reason=plan.name)

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

    def default_moves(self) -> str:
        """Stage 2 default: canonicalization's fusion stage and the stages after it."""
        fuse_and_finish(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    def finish_moves(self) -> str:
        """The stages after fusion, once :meth:`apply_move` chose the granularity by hand."""
        finish(self.sdfg, self.targets)
        narrow_nested_memlets(self.sdfg)
        self.bump()
        return self.describe()

    # Stage 3: scopes

    def default_scopes(self) -> list[KernelScope]:
        """Stage 3 default: one kernel per parallel top-level map; returns the new kernels."""
        lowered = lower_nests_to_external_call(self.sdfg)
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

    def set_scope(self, labels: Sequence[str], epoch: int) -> MoveResult:
        """Make one kernel of the regions ``labels`` name: several top-level maps of one state, or a straight run of
        blocks of one control-flow region, at any depth. The kernel written in stage 5 fuses what no move could.

        :param labels: Tree labels of the maps, or of the blocks, as :meth:`describe` printed them at ``epoch``.
        :param epoch: The epoch those labels were read at.
        :returns: ``applied`` with the new kernel's name as ``reason``, or why nothing changed.
        """
        names = tuple(labels)
        rows = self.named_rows("define-scope", names, epoch)
        if isinstance(rows, MoveResult):
            return rows
        lowered = lower_group_to_external_call(self.sdfg, rows)
        if isinstance(lowered, str):
            return MoveResult(status=MoveStatus.ILLEGAL, kind="define-scope", labels=names, reason=lowered)
        self.unplaced = None
        self.bump()
        return MoveResult(status=MoveStatus.APPLIED, kind="define-scope", labels=names, reason=lowered[0].name)

    def unset_scope(self, name: str, epoch: int) -> MoveResult:
        """Inline the kernel ``name`` back into the program: the inverse of :meth:`set_scope`. A placement is
        dropped first, as a new :meth:`place` would; its kernel source and tuning are dropped with the kernel.

        :returns: ``applied`` with ``name`` as ``reason``, or why nothing changed.
        """
        if epoch != self.epoch:
            reason = f"kernels read at epoch {epoch}; the program is at epoch {self.epoch}."
            return MoveResult(status=MoveStatus.STALE, kind="unset-scope", labels=(name,), reason=reason)
        calls = {ext.name: ext for ext in external_calls(self.sdfg)}
        if name not in calls:
            reason = f"no kernel is named {name}; kernels are {', '.join(sorted(calls))}."
            return MoveResult(status=MoveStatus.NOT_FOUND, kind="unset-scope", labels=(name,), reason=reason)
        if calls[name].standalone_sdfg is None:
            reason = f"{name} keeps no SDFG of its nest to inline."
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
        self.bump()
        return MoveResult(status=MoveStatus.APPLIED, kind="unset-scope", labels=(name,), reason=name)

    def kernel_graph(self) -> KernelGraph:
        """What can reach every kernel argument and program output (:func:`kernel_dependencies`), once per epoch."""
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

    def place(self, devices: dict[str, str] | None = None, epoch: int | None = None) -> PlaceResult:
        """Run each kernel on a device and insert the host/device copies. Kernels ``devices`` leaves out keep the
        default: parallel kernels on the GPU when it is a target, the rest on the CPU. A later call starts again
        from the program before the first placement.

        :param devices: Kernel name -> ``cpu`` or ``gpu``.
        :param epoch: The epoch the kernel names were read at; required with ``devices``.
        """
        if devices is not None and epoch != self.epoch:
            return PlaceResult(
                status=MoveStatus.STALE, reason=f"kernels read at epoch {epoch}; the program is at {self.epoch}."
            )
        chosen = default_devices(self.sdfg, self.targets) | (devices or {})
        refusal = check_devices(self.sdfg, self.targets, chosen)
        if refusal is not None:
            return PlaceResult(status=MoveStatus.ILLEGAL, reason=refusal)
        if self.unplaced is not None:
            self.sdfg = copy.deepcopy(self.unplaced)
            self.kernel_deps = None
        try:
            moves = tuple(transfers(self.kernel_graph(), chosen))
        except UnsupportedProgram:
            moves = None
        self.unplaced = copy.deepcopy(self.sdfg)
        placement = place(self.sdfg, self.targets, chosen)
        self.bump()
        return PlaceResult(
            status=MoveStatus.APPLIED, devices=dict(placement.devices), copies=tuple(placement.copies), transfers=moves
        )

    # Stage 5: kernels

    def default_kernel(self, name: str) -> KernelSource:
        """The kernel's CPF unit for its device, rendered once per epoch."""
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
        use_kernel_library(ext, build.archive, src, runtime)
        self.builds[name] = build
        self.measured = None

    def optimize_kernel(self, name: str) -> KernelLibrary:
        """Stage 5 default: render the kernel's CPF unit, build it with the first configuration stage 6 sweeps for
        its device, and link it into the kernel's ``ExternalCall``."""
        src = self.default_kernel(name)
        variant = self.first_variant(src.device)
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

    def kernel_source(self, name: str) -> str:
        """The source of the kernel's current library, or of its default CPF unit before one is built."""
        build = self.builds.get(name)
        return kernel_text(build.source if build is not None else self.default_kernel(name))

    def set_kernel_source(self, name: str, source: str, language: str, reps: int = TIMED_REPS) -> KernelOutcome:
        """Build a kernel source with the kernel's entry, validate it against the kernel's Python oracle, and link it
        when it matches.

        :param source: A translation unit defining ``extern "C" void <name>(...)`` with the parameters, in order, of
            the default unit (:meth:`kernel_source` before any change). A GPU kernel may be two units, host then
            device, split by the line ``DEVICE_UNIT_MARKER``, as :meth:`kernel_source` returns it.
        :param language: ``cpp`` (C++ with OpenMP) for a CPU kernel, ``cuda`` for a GPU kernel.
        :returns: The outcome; only ``ok`` changes the kernel's library.
        """
        default = self.default_kernel(name)
        if language != LANGUAGES[default.device]:
            return KernelOutcome(
                kernel=name,
                status=KernelStatus.REFUSED,
                reason=f"a {default.device} kernel is {LANGUAGES[default.device]}.",
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
        suffix = FORMS[default.device].suffix
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
        try:
            archive = build_kernel_library(src, variant.compiler, list(variant.flags), attempt)
        except RuntimeError as err:
            return KernelOutcome(kernel=name, status=KernelStatus.BUILD_FAILED, reason=str(err)[-ERROR_TAIL:])
        verdict = validate_kernel(
            archive, src, self.prepare_kernel(name), self.kernel_sizes(name), reps, variant.fp_mode
        )
        if not verdict.ok:
            reason = verdict.error or f"max rel err {verdict.md_rel:.3g} at {verdict.fp_mode}"
            return KernelOutcome(kernel=name, status=KernelStatus.WRONG, reason=reason)
        self.bind(name, KernelBuild(src, variant, archive, verdict))
        return KernelOutcome(kernel=name, status=KernelStatus.OK, time_us=verdict.time_us)

    def prepare_kernel(self, name: str) -> Prepared:
        if name not in self.prepared:
            self.prepared[name] = prepare(node_boundary(self.kernel(name)), name)
        return self.prepared[name]

    def need_sizes(self) -> dict[str, int]:
        if not self.sizes:
            raise ValueError("this session has no sizes; pass Session(..., sizes={symbol: value})")
        return self.sizes

    def kernel_sizes(self, name: str) -> dict[str, int]:
        """The sizes plus, for a kernel inside a sequential loop, a middle value of each loop iterator it takes;
        a value given in ``sizes`` wins."""
        sizes = self.need_sizes()
        ext = self.kernel(name)
        return {**loop_symbol_values(find_state_of_node(self.sdfg, ext), sizes), **sizes}

    # Stage 6: variants

    def sweep(self, name: str, reps: int = TIMED_REPS, compilers: list[str] | None = None) -> Sweep:
        """Build and time the kernel's current source under every configuration of its device, link the fastest
        correct one, and summarize the sweep; ``config`` is the winner's compiler, FP mode, cost model, flags, time.

        :param compilers: Toolchain names to keep (``gcc``, ``clang``, ``nvcc``, ``nvcc-13.1``, ...); every one
            discovered for the kernel's device when ``None``.
        """
        build = self.builds.get(name)
        src = build.source if build is not None else self.default_kernel(name)
        variants = device_variants(src.device, compilers)
        prep = self.prepare_kernel(name)
        result = select_variant(src, prep, self.kernel_sizes(name), reps, variants, self.work_dir / name / "variants")
        winner = result.winner
        if winner is not None and winner.archive is not None:
            self.bind(name, KernelBuild(src, winner.variant, winner.archive, winner.verdict))
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
                self.optimize_kernel(ext.name)
            build = self.builds[ext.name]
            if build.verdict is None:
                sizes, fp_mode = self.kernel_sizes(ext.name), build.variant.fp_mode
                computed = [s for s in build.source.boundary.symbols if s not in sizes]
                build.verdict = (
                    unchecked_verdict(fp_mode, computed)
                    if computed
                    else validate_kernel(
                        build.archive, build.source, self.prepare_kernel(ext.name), sizes, reps, fp_mode
                    )
                )
            variant = build.variant
            remarks = feedback.compiler_remarks(build.source, variant.compiler, list(variant.flags))
            evidence.append(feedback.Evidence(ext.name, build.verdict, remarks))
        return evidence

    def trial(self) -> Session:
        """A copy of this session in its own work directory, at the same stage, for defaults that must not stick."""
        copied = Session(copy.deepcopy(self.sdfg), self.targets, self.name, sizes=self.sizes)
        # same epoch, same labels: what was read on this session names the same rows of the copy
        copied.epoch, copied.unplaced = self.epoch, copy.deepcopy(self.unplaced)
        # the copied kernels link this session's libraries, so the trial measures them instead of rebuilding defaults
        copied.sources, copied.builds = dict(self.sources), dict(self.builds)
        return copied

    def performance(
        self, inputs: dict[str, np.ndarray], reference: Reference | None = None, reps: int = TIMED_REPS
    ) -> Performance:
        """Every phase's feedback at once (:mod:`~nestforge.stages.performance`): on a :meth:`trial`, the default
        scopes when none is defined, the default placement when none is chosen and the default kernel builds, then
        the whole program (checked against ``reference`` when given), its offloading and each kernel are timed. Reused
        until the program changes."""
        if self.measured is not None:
            return self.measured
        trial = self.trial()
        if not external_calls(trial.sdfg):
            trial.default_scopes()
        devices = {ext.name: kernel_device(ext) for ext in external_calls(trial.sdfg)}
        if trial.unplaced is None:
            placed = trial.place()
            devices, moves = placed.devices, placed.transfers
        else:
            try:
                moves = tuple(transfers(trial.kernel_graph(), devices))
            except UnsupportedProgram:
                moves = None
        evidence = trial.kernel_evidence(reps)
        program = time_program(
            trial.sdfg, trial.work_dir / "program", inputs, reference, self.sizes, reps, "gpu" in devices.values()
        )
        # the trial's builds are measured; a long session would otherwise keep every one until it exits
        shutil.rmtree(trial.work_dir, ignore_errors=True)
        self.measured = Performance(
            structured=program,
            offloading=offload_report(trial.sdfg, devices, moves, self.sizes),
            kernels=kernel_reports(evidence, devices),
            hints=feedback.report(evidence),
        )
        return self.measured
