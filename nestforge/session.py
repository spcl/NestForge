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
import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import dace
from dace.libraries.standard.nodes.external_call import ExternalCall, external_calls
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.build.arena import TIMED_REPS
from nestforge.build.toolchain import parse_params, raw_signature
from nestforge.corpus.translate import Prepared, prepare
from nestforge.ir.depends import OUTPUT_PREFIX, KernelGraph, UnsupportedProgram, kernel_dependencies
from nestforge.ir.introspect import Row, describe_graph, tree_rows
from nestforge.ir.loops import kernel_state, loop_symbol_values
from nestforge.ir.names import normalize_labels
from nestforge.stages import feedback
from nestforge.stages.canonicalize import Targets, canonicalize, finish, fuse_and_finish
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
    use_kernel_library,
    validate_kernel,
)
from nestforge.stages.moves import (
    MOVE_SHAPES,
    NOT_IMPLEMENTED,
    Rewrite,
    check_kind,
    legal_moves,
    plan_move,
    scope_metrics,
    sdfg_metrics,
)
from nestforge.stages.placement import check_devices, default_devices, kernel_device, place, transfers
from nestforge.stages.scopes import (
    is_parallel_kernel,
    lower_group_to_external_call,
    lower_nests_to_external_call,
    node_boundary,
)
from nestforge.stages.variants import Variant, VariantCell, device_variants, select_variant

#: Spaces per level of the saved JSON snapshots.
JSON_INDENT = 2

#: Characters of a failed build's error text a caller gets back.
ERROR_TAIL = 1500

#: The kernel source language each device's kernels are written in.
LANGUAGES = {"cpu": "cpp", "gpu": "cuda"}


@dataclass(frozen=True, slots=True)
class MoveResult:
    """What a labeled call did. ``status`` is ``applied``, ``illegal``, ``not-implemented``, ``not-found`` or
    ``stale``; only ``applied`` changed the program, and ``reason`` then names what ran."""

    status: str
    kind: str
    labels: tuple[str, ...]
    reason: str


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
        "work_dir",
        "kernel_deps",
        "unplaced",
        "sources",
        "builds",
        "prepared",
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
        self.work_dir = Path(work_dir) if work_dir else scratch_dir()
        self.kernel_deps: KernelGraph | None = None
        # the program before stage 4, so a new placement starts from it
        self.unplaced: dace.SDFG | None = None
        self.sources: dict[str, KernelSource] = {}
        self.builds: dict[str, KernelBuild] = {}
        self.prepared: dict[str, Prepared] = {}

    def bump(self) -> None:
        self.epoch += 1
        normalize_labels(self.sdfg)
        self.rows = None
        self.kernel_deps = None
        self.sources, self.builds, self.prepared = {}, {}, {}

    def row_index(self) -> dict[str, Row]:
        """Tree label -> ``(block or node, state)``, built once per epoch."""
        if self.rows is None:
            self.rows = tree_rows(self.sdfg)
        return self.rows

    def named_rows(self, kind: str, names: tuple[str, ...], epoch: int) -> list[Row] | MoveResult:
        """The rows ``names`` label at ``epoch``, or the ``stale`` or ``not-found`` result."""
        if epoch != self.epoch:
            reason = f"labels read at epoch {epoch}; the program is at epoch {self.epoch}. Describe again."
            return MoveResult("stale", kind, names, reason)
        rows = self.row_index()
        missing = [name for name in names if name not in rows]
        if missing:
            return MoveResult("not-found", kind, names, f"no tree row is labeled {', '.join(missing)}.")
        return [rows[name] for name in names]

    # Stage 1: canonicalize

    def canonicalize(self) -> str:
        """Canonicalize up to the fusion stage for the session's targets; returns the new tree."""
        canonicalize(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    # Stage 2: moves

    def describe(self, bodies: bool = False, metrics: bool = False, deps: bool = False) -> str:
        """The program as a text tree headed by the epoch; rows are named by the labels the labeled calls take.
        ``metrics`` appends each top-level map's work, depth, bytes and OI; ``deps`` puts each kernel's
        :meth:`kernel_graph` line under its row."""
        return describe_graph(
            self.sdfg,
            bodies=bodies,
            metrics=(lambda entry: scope_metrics(self.sdfg, entry).suffix()) if metrics else None,
            notes=self.deps_line if deps else None,
            epoch=self.epoch,
        )

    def deps_line(self, node: nodes.LibraryNode) -> str | None:
        graph = self.kernel_graph()
        return graph.line(node.label) if node.label in graph.kernels else None

    def list_moves(self, kind: str | None = None) -> list[dict]:
        """Every legal move right now, of ``kind`` or of every kind, as ``{kind, labels, epoch}``: exactly what
        :meth:`apply_move` takes back. A not-implemented kind lists nothing."""
        return [{"kind": k, "labels": list(labels), "epoch": self.epoch} for k, labels in legal_moves(self.sdfg, kind)]

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
            return MoveResult("not-implemented" if kind in NOT_IMPLEMENTED else "illegal", kind, names, plan)
        plan.commit()
        self.unplaced = None
        self.bump()
        return MoveResult("applied", kind, names, plan.name)

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
        except TypeError as err:
            return str(err)

    def default_moves(self) -> str:
        """Stage 2 default: canonicalization's fusion stage and the stages after it."""
        fuse_and_finish(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    def finish_moves(self) -> str:
        """The stages after fusion, once :meth:`apply_move` chose the granularity by hand."""
        finish(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    # Stage 3: scopes

    def define_scopes(self) -> list[dict]:
        """Stage 3 default: one kernel per parallel top-level map; returns the new kernels."""
        lowered = lower_nests_to_external_call(self.sdfg)
        if lowered:
            self.unplaced = None
            self.bump()
            self.snapshot_kernel_graph()
        return [
            {
                "name": ext.name,
                "reads": list(boundary.inputs),
                "writes": list(boundary.outputs),
                "symbols": list(boundary.symbols),
                "parallel": is_parallel_kernel(ext),
            }
            for ext, boundary in lowered
        ]

    def define_scope(self, labels: Sequence[str], epoch: int) -> MoveResult:
        """Make one kernel of the regions ``labels`` name: several top-level maps of one state, or a straight run of
        top-level blocks. The kernel written in stage 5 fuses what no move could.

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
            return MoveResult("illegal", "define-scope", names, lowered)
        self.unplaced = None
        self.bump()
        self.snapshot_kernel_graph()
        return MoveResult("applied", "define-scope", names, lowered[0].name)

    def kernel_graph(self) -> KernelGraph:
        """What can reach every kernel argument and program output (:func:`kernel_dependencies`), once per epoch."""
        if self.kernel_deps is None:
            self.kernel_deps = kernel_dependencies(self.sdfg)
        return self.kernel_deps

    def snapshot_kernel_graph(self) -> None:
        """Save :meth:`kernel_graph` to ``<work_dir>/kernel_deps/e<epoch>.json``, byte-stable, when the analysis
        accepts the program."""
        try:
            graph = self.kernel_graph()
        except UnsupportedProgram:
            return
        path = self.work_dir / "kernel_deps" / f"e{self.epoch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(graph.to_json(), indent=JSON_INDENT, sort_keys=True) + "\n")

    def kernel(self, name: str) -> ExternalCall:
        """The kernel node named ``name``."""
        calls = {ext.name: ext for ext in external_calls(self.sdfg)}
        if name not in calls:
            raise KeyError(f"no kernel is named {name!r}; kernels are {sorted(calls)}")
        return calls[name]

    def list_kernels(self) -> list[dict]:
        """Every kernel with its device, parallel flag, arguments, the producer labels reaching each argument
        (``depends``), and the loops a reaching value crossed (``carried``)."""
        graph = self.kernel_graph()
        out: list[dict] = []
        for name in graph.kernels:
            ext, arguments = self.kernel(name), graph.arguments(name)
            out.append(
                {
                    "name": name,
                    "device": kernel_device(ext),
                    "parallel": is_parallel_kernel(ext),
                    "inputs": [edge.arg for edge in arguments if edge.role == "input"],
                    "outputs": sorted(conn.removeprefix(OUTPUT_PREFIX) for conn in ext.out_connectors),
                    "symbols": [edge.arg for edge in arguments if edge.role == "symbol"],
                    "depends": {edge.arg: edge.labels() for edge in arguments},
                    "carried": {edge.arg: edge.loops() for edge in arguments if edge.loops()},
                }
            )
        return out

    # Stage 4: placement

    def place(self, devices: dict[str, str] | None = None, epoch: int | None = None) -> dict:
        """Run each kernel on a device and insert the host/device copies. Kernels ``devices`` leaves out keep the
        default: parallel kernels on the GPU when it is a target, the rest on the CPU. A later call starts again
        from the program before the first placement.

        :param devices: Kernel name -> ``cpu`` or ``gpu``.
        :param epoch: The epoch the kernel names were read at; required with ``devices``.
        :returns: ``status`` and ``reason`` as :class:`MoveResult` has them; once ``applied``, every kernel's
            device, the copies placed, and the transfers the kernel DAG implies (``None`` when the analysis refuses
            the program).
        """
        if devices is not None and epoch != self.epoch:
            return {"status": "stale", "reason": f"kernels read at epoch {epoch}; the program is at {self.epoch}."}
        chosen = default_devices(self.sdfg, self.targets) | (devices or {})
        refusal = check_devices(self.sdfg, self.targets, chosen)
        if refusal is not None:
            return {"status": "illegal", "reason": refusal}
        if self.unplaced is not None:
            self.sdfg = copy.deepcopy(self.unplaced)
            self.kernel_deps = None
        try:
            moves = [asdict(move) for move in transfers(self.kernel_graph(), chosen)]
        except UnsupportedProgram:
            moves = None
        self.unplaced = copy.deepcopy(self.sdfg)
        placement = place(self.sdfg, self.targets, chosen)
        self.bump()
        return {
            "status": "applied",
            "reason": "",
            "kernels": [{"name": name, "device": device} for name, device in placement.devices.items()],
            "copies": [list(pair) for pair in placement.copies],
            "transfers": moves,
        }

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

    def optimize_kernel(self, name: str) -> dict:
        """Stage 5 default: render the kernel's CPF unit, build it with the first configuration stage 6 sweeps for
        its device, and link it into the kernel's ``ExternalCall``."""
        src = self.default_kernel(name)
        variant = self.first_variant(src.device)
        archive = build_kernel_library(src, variant.compiler, list(variant.flags), self.work_dir / name / "library")
        self.bind(name, KernelBuild(src, variant, archive))
        return {
            "kernel": name,
            "symbol": src.symbol,
            "abi_order": list(src.abi_order),
            "entry": f"{src.symbol}({', '.join(src.abi_order)})",
            "unit": str(src.unit),
            "library": str(archive),
            "variant": variant.label,
        }

    def kernel_source(self, name: str) -> str:
        """The source of the kernel's current library, or of its default CPF unit before one is built."""
        build = self.builds.get(name)
        return kernel_text(build.source if build is not None else self.default_kernel(name))

    def set_kernel_source(self, name: str, source: str, language: str, reps: int = TIMED_REPS) -> dict:
        """Build a kernel source with the kernel's entry, validate it against the kernel's Python oracle, and link it
        when it matches.

        :param source: A translation unit defining ``extern "C" void <name>(...)`` with the parameters, in order, of
            the default unit (:meth:`kernel_source` before any change). A GPU kernel may be two units, host then
            device, split by the line ``DEVICE_UNIT_MARKER``, as :meth:`kernel_source` returns it.
        :param language: ``cpp`` (C++ with OpenMP) for a CPU kernel, ``cuda`` for a GPU kernel.
        :returns: ``status`` (``ok``, ``refused``, ``build-failed`` or ``wrong``), ``reason`` and ``time_us``; only
            ``ok`` changes the kernel's library.
        """
        default = self.default_kernel(name)
        if language != LANGUAGES[default.device]:
            return self.kernel_outcome(name, "refused", f"a {default.device} kernel is {LANGUAGES[default.device]}.")
        try:
            params = [p.name for p in parse_params(raw_signature(source, name))]
        except (LookupError, ValueError) as err:
            return self.kernel_outcome(name, "refused", str(err))
        if params != default.abi_order:
            return self.kernel_outcome(name, "refused", f"the entry takes {params}; it must take {default.abi_order}.")
        attempt = self.work_dir / name / f"agent{len(list((self.work_dir / name).glob('agent*')))}"
        attempt.mkdir(parents=True)
        host, device_text = split_kernel_text(source)
        suffix = FORMS[default.device].suffix
        unit = attempt / f"{name}{suffix}"
        unit.write_text(host)
        device_unit = None if device_text is None else attempt / f"{name}{DEVICE_UNIT_SUFFIX}{suffix}"
        if device_unit is not None:
            device_unit.write_text(device_text)
        src = KernelSource(name, unit, list(default.abi_order), default.boundary, default.device, device_unit)
        variant = self.first_variant(src.device)
        try:
            archive = build_kernel_library(src, variant.compiler, list(variant.flags), attempt)
        except RuntimeError as err:
            return self.kernel_outcome(name, "build-failed", str(err)[-ERROR_TAIL:])
        verdict = validate_kernel(
            archive, src, self.prepare_kernel(name), self.kernel_sizes(name), reps, variant.fp_mode
        )
        if not verdict.ok:
            reason = verdict.error or f"max rel err {verdict.md_rel:.3g} at {verdict.fp_mode}"
            return self.kernel_outcome(name, "wrong", reason)
        self.bind(name, KernelBuild(src, variant, archive, verdict))
        return self.kernel_outcome(name, "ok", "", verdict.time_us)

    @staticmethod
    def kernel_outcome(name: str, status: str, reason: str, time_us: float | None = None) -> dict:
        return {"kernel": name, "status": status, "reason": reason, "time_us": time_us}

    def prepare_kernel(self, name: str) -> Prepared:
        if name not in self.prepared:
            self.prepared[name] = prepare(node_boundary(self.kernel(name)), name, self.work_dir / name)
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
        return {**loop_symbol_values(kernel_state(self.sdfg, ext), sizes), **sizes}

    # Stage 6: variants

    def sweep(self, name: str, reps: int = TIMED_REPS, compilers: list[str] | None = None) -> dict:
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
        return {
            "kernel": name,
            "cells": len(result.cells),
            "collapsed": list(result.collapsed),
            "winner": winner.variant.label if winner is not None else None,
            "config": winner_config(winner),
        }

    # Stage 7: feedback

    def feedback(self, reps: int = TIMED_REPS) -> str:
        """A few lines of hints for the next round, most important first, from every kernel's current library:
        its validation and time, and the compiler's vectorizer or register remarks."""
        evidence: list[feedback.Evidence] = []
        for ext in external_calls(self.sdfg):
            if ext.name not in self.builds:
                self.optimize_kernel(ext.name)
            build = self.builds[ext.name]
            if build.verdict is None:
                prep = self.prepare_kernel(ext.name)
                fp_mode = build.variant.fp_mode
                build.verdict = validate_kernel(
                    build.archive, build.source, prep, self.kernel_sizes(ext.name), reps, fp_mode
                )
            variant = build.variant
            remarks = feedback.compiler_remarks(build.source, variant.compiler, list(variant.flags))
            evidence.append(feedback.Evidence(ext.name, build.verdict, remarks))
        return feedback.report(evidence)


def winner_config(winner: VariantCell | None) -> dict:
    """The configuration stage 6 chose: compiler, FP mode, cost model, flags and measured time, or all ``None``."""
    if winner is None:
        return dict.fromkeys(("compiler", "fp_mode", "cost_model", "flags", "time_us"))
    variant = winner.variant
    return {
        "compiler": variant.toolchain,
        "fp_mode": variant.fp_mode,
        "cost_model": variant.cost_model,
        "flags": list(variant.flags),
        "time_us": winner.verdict.time_us,
    }
