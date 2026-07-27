# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loopnest -> winning-config dataset every sweep leaves behind.

A sweep already knows, per nest, which configuration won and by how much; it prints that in a table and
then throws it away. Appending it here instead accumulates a (nest features, winning config) corpus across
every job ever run -- the training set for predicting the winner instead of searching for it.

Append-only JSONL, one record per (nest, axis, job). JSONL because jobs are concurrent (one rank per
socket, several nodes) and an append of a single short line is atomic on POSIX up to PIPE_BUF, so ranks
share one file without a lock or a merge step. A rewritten-whole JSON would need both.

The nest is identified by :func:`~nestforge.dedup.cpp_body_key` over its emitted C -- flags never reach
the emitted source, so the SAME nest keeps ONE identity across compilers, FP rungs and machines, and the
dataset can ask "what won for this loopnest elsewhere?". A kernel key cannot do that: `s000` is a
different nest per opt-mode, and two corpora spell the same computation under different names.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import dace
import numpy as np

from nestforge.dedup import cpp_body_key

#: Where records land when a caller names no directory. Relative, so a job writes under its own result
#: root rather than a path baked in here; ``NF_DATASET_DIR`` overrides it.
DATASET_DIR: str = os.environ.get("NF_DATASET_DIR", ".results")

#: One file per axis keeps a reader from filtering, and keeps two concurrent sweeps of different axes off
#: each other's lines entirely.
FILENAME: str = "winners-{axis}.jsonl"


@dataclass(slots=True)
class WinnerRecord:
    """One nest, one search axis, the configuration that won it, and what winning was worth."""
    kernel: str
    corpus: str
    nest: int
    fingerprint: str  # cpp_body_key of the emitted nest: identity independent of flags/compiler/machine
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
    from nestforge.vectorize_variants import has_integer_conditional_write, has_same_write_set_branch

    maps, tasklets, innermost, symbolic = 0, 0, [], 0
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
            extent = str((end - begin + 1) / step)
            innermost.append(extent)
            if not extent.lstrip("-").isdigit():
                symbolic += 1
    arrays = [d for d in sdfg.arrays.values() if not d.transient]
    return {
        "maps":
        maps,
        "tasklets":
        tasklets,
        "innermost_maps":
        len(innermost),
        "symbolic_extents":
        symbolic,  # a symbolic extent means the remainder is ALWAYS emitted
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


def fingerprint(c_source: Optional[Path]) -> str:
    """The nest's flag-independent identity, or ``""`` when its source is unavailable.

    Empty rather than a raise: a record with no fingerprint still carries features and a winner, and
    losing the whole record would be the larger loss. A reader joins on a non-empty fingerprint."""
    if c_source is None or not c_source.is_file():
        return ""
    return cpp_body_key(c_source.read_text())


def record_winner(record: WinnerRecord, out_dir: Optional[Path] = None) -> Path:
    """Append ``record`` to ``<out_dir>/winners-<axis>.jsonl`` and return the file.

    Never raises on a full/unwritable directory: a sweep that measured for an hour must not lose its
    tables because the dataset could not be appended to. The failure goes to the record file's directory
    being absent from the caller's results instead."""
    out = Path(out_dir if out_dir is not None else DATASET_DIR)
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
    out = Path(out_dir if out_dir is not None else DATASET_DIR)
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
