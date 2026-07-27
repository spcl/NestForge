# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loopnest -> winning-config dataset every sweep leaves behind.

A sweep already knows, per nest, which configuration won and by how much; it prints that in a table and
then throws it away. Appending it here instead accumulates a (nest features, winning config) corpus across
every job ever run -- the training set for predicting the winner instead of searching for it.

Append-only JSONL, one record per (nest, axis, job). JSONL because jobs are concurrent (one rank per
socket, several nodes) and an append of a single short line is atomic on POSIX up to PIPE_BUF, so ranks
share one file without a lock or a merge step. A rewritten-whole JSON would need both.

The nest is identified by its NUMPY source, and stored by it. Every emitted variant -- C, C++, Fortran --
is a translation of that one rendering (:func:`nestforge.emit_numpy.nest_to_numpy`), so it is the only
form that is both upstream of the language axis and independent of compiler, flags and machine. Keying on
the emitted C instead tied a nest's identity to one lane: a Fortran-only run recorded no identity at all.
A kernel key cannot do the job either -- `s000` is a different nest per opt-mode, and two corpora spell
the same computation under different names.

The source itself is content-addressed under ``nests/<fingerprint>.py`` rather than copied into every
record: a nest measured by a hundred jobs is one file, and a record stays a row rather than a document.
That also makes a recorded winner reproducible -- the exact input the translator consumed is on disk next
to it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import dace
import numpy as np
from dace import symbolic

from nestforge.vectorize_variants import has_integer_conditional_write, has_same_write_set_branch

#: Where records land when a caller names no directory. Relative, so a job writes under its own result
#: root rather than a path baked in here.
DEFAULT_DIR: str = ".results"


def dataset_dir(out_dir: Optional[Path] = None) -> Path:
    """``out_dir``, else ``$NF_DATASET_DIR``, else :data:`DEFAULT_DIR`.

    Resolved per call rather than captured at import, so the env var actually overrides -- an import-time
    constant answers to whatever the environment was when the first module happened to import this one."""
    if out_dir is not None:
        return Path(out_dir)
    return Path(os.environ.get("NF_DATASET_DIR") or DEFAULT_DIR)


#: One file per axis keeps a reader from filtering, and keeps two concurrent sweeps of different axes off
#: each other's lines entirely.
FILENAME: str = "winners-{axis}.jsonl"

#: Content-addressed nest store: the numpy rendering every record's ``fingerprint`` points at.
NEST_DIR: str = "nests"


@dataclass(slots=True)
class WinnerRecord:
    """One nest, one search axis, the configuration that won it, and what winning was worth."""
    kernel: str
    corpus: str
    nest: int
    fingerprint: str  # sha of the numpy nest's AST: one identity across languages, flags and machines
    axis: str  # which search produced this winner ("vectorize", "flags", "granularity", ...)
    winner: Dict[str, object]  # the winning configuration, as the axis spells it
    median_us: float
    baseline_us: Optional[float] = None  # what the winner is measured against, when the sweep has one
    speedup: Optional[float] = None
    features: Dict[str, object] = field(default_factory=dict)
    compiler: str = ""
    machine: str = field(default_factory=platform.machine)
    host: str = field(default_factory=platform.node)
    timestamp: float = field(default_factory=time.time)  # epoch seconds; jobs land out of order


def has_fma_pattern(sdfg: dace.SDFG) -> bool:
    """True when the nest contains both a multiply and an add, so an FMA can form in it.

    A FEATURE, not a knob: the vectorizer's ``fuse_multiply_add`` follows the FP rung
    (:func:`~nestforge.vectorize_variants.fma_allowed`), but whether a nest HAS something to fuse is
    exactly the kind of structure a model would key a prediction on.

    Checked across the whole nest rather than per tasklet: ``SplitTasklets`` runs inside the vectorizer and
    can split ``a*b + c`` into two tasklets that the FMA pass then re-fuses."""
    ops = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet) and node.code is not None:
                text = node.code.as_string or ""
                ops.update(op for op in ("*", "+") if op in text)
    return {"*", "+"} <= ops


def nest_features(sdfg: dace.SDFG) -> Dict[str, object]:
    """The structural features of a nest, as a model would see it -- shape and content only, never a
    timing or a config, so a record's features are the INPUT and its winner is the label.

    Deliberately cheap and syntactic: everything here reads the SDFG that is already in memory, so
    recording costs a graph walk against a sweep that already spent minutes compiling."""
    maps, tasklets, innermost, symbolic_count = 0, 0, [], 0
    for state in sdfg.all_states():
        scope = state.scope_children()
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                tasklets += 1
            if not isinstance(node, dace.nodes.MapEntry):
                continue
            maps += 1
            if any(isinstance(child, dace.nodes.MapEntry) for child in scope[node]):
                continue
            begin, end, step = node.map.range[-1]
            # int_ceil, never `/`: true division renders a step-2 range as "3.5", which then fails the
            # isdigit() test and books a fully CONSTANT extent as symbolic -- the feature whose whole
            # meaning is "the remainder is always emitted".
            extent = symbolic.symstr(symbolic.int_ceil(end - begin + 1, step))
            innermost.append(extent)
            if not extent.lstrip("-").isdigit():
                symbolic_count += 1
    arrays = [d for d in sdfg.arrays.values() if not d.transient]
    return {
        "maps":
        maps,
        "tasklets":
        tasklets,
        "innermost_maps":
        len(innermost),
        "symbolic_extents":
        symbolic_count,  # a symbolic extent means the remainder is ALWAYS emitted
        "innermost_extents":
        innermost,
        "arrays":
        len(arrays),
        # numpy spelling ("float64"), not dace's C name ("double"): the dataset is read by numpy/pandas,
        # and the same nest must not key differently because a descriptor renders its dtype for C.
        "dtypes":
        sorted({np.dtype(d.dtype.type).name
                for d in arrays}),
        "max_map_dims":
        max((len(n.map.params) for s in sdfg.all_states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)),
            default=0),
        "has_branch":
        has_same_write_set_branch(sdfg),
        "has_integer_conditional":
        has_integer_conditional_write(sdfg),
        "has_fma_pattern":
        has_fma_pattern(sdfg),
    }


def fingerprint(numpy_source: Optional[str]) -> str:
    """The nest's identity: a hash of its numpy source, normalized through the AST.

    Through the AST rather than over the text, because a rename of the emitted function or a change in the
    renderer's whitespace is not a different loopnest -- and the dataset's whole value is that the same
    nest, met again in another job, joins to what won for it last time. Unparseable source falls back to
    the raw text (a hash that over-splits costs a join; one that over-merges corrupts the labels).

    ``""`` when there is no source: a record with no identity still carries features and a winner, and
    losing the whole record would be the larger loss. A reader joins on a NON-empty fingerprint."""
    if not numpy_source:
        return ""
    try:
        canonical = ast.dump(ast.parse(numpy_source), annotate_fields=True, include_attributes=False)
    except SyntaxError:
        canonical = numpy_source
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def store_nest(numpy_source: str, out_dir: Optional[Path] = None) -> str:
    """Write ``numpy_source`` to ``<out_dir>/nests/<fingerprint>.py`` if it is not already there, and
    return the fingerprint. Content-addressed, so concurrent ranks meeting the same nest write the same
    bytes to the same path and the last writer wins harmlessly."""
    key = fingerprint(numpy_source)
    if not key:
        return ""
    out = dataset_dir(out_dir) / NEST_DIR
    path = out / f"{key}.py"
    try:
        if not path.exists():
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(numpy_source)
    except OSError:
        pass  # a by-product; the record still carries the fingerprint and the features
    return key


def load_nest(key: str, out_dir: Optional[Path] = None) -> Optional[str]:
    """The numpy source behind a record's ``fingerprint``, or ``None`` when this results tree never saw it."""
    path = dataset_dir(out_dir) / NEST_DIR / f"{key}.py"
    return path.read_text() if path.is_file() else None


def record_winner(record: WinnerRecord, out_dir: Optional[Path] = None) -> Path:
    """Append ``record`` to ``<out_dir>/winners-<axis>.jsonl`` and return the file.

    Never raises on a full/unwritable directory: a sweep that measured for an hour must not lose its
    tables because the dataset could not be appended to. The failure goes to the record file's directory
    being absent from the caller's results instead."""
    out = dataset_dir(out_dir)
    path = out / FILENAME.format(axis=record.axis)
    try:
        out.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")
    except OSError:
        pass
    return path


def load_records(out_dir: Optional[Path] = None, axis: Optional[str] = None) -> List[Dict]:
    """Every record under ``out_dir``, optionally for one axis. A truncated final line (a job killed
    mid-append) is skipped, not fatal: the point of an append-only log is that what landed stays readable.
    """
    out = dataset_dir(out_dir)
    pattern = FILENAME.format(axis=axis) if axis else FILENAME.format(axis="*")
    rows: List[Dict] = []
    for path in sorted(out.glob(pattern)):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
