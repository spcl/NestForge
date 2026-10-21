# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers several test modules share."""

from __future__ import annotations

import copy
import ctypes
import inspect
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dace
import numpy as np
import yaml
from dace import symbolic
from dace.libraries.standard.nodes.external_call import ExpandExternCall, ExternalCall
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState

from nestforge.build import toolchain
from nestforge.build.sdfg import (
    BuildOptions,
    BuiltSDFG,
    build_archive,
    build_commands,
    generate_program,
    link_shared,
)
from nestforge.build.toolchain import CType, parse_params, raw_signature, signature
from nestforge.corpus import bench
from nestforge.corpus.bench import CorpusKernel
from nestforge.corpus.translate import prepare
from nestforge.ir.emit_python import (
    kernel_args,
    kernel_arrays,
    load_emitted,
    lower,
    oracle_sdfg,
    render,
    scratch_arrays,
    widen_scratch,
)
from nestforge.ir.extract import Boundary, CfgNest, NestNode, extract_cfg_nest, extract_map_nest
from nestforge.ir.introspect import body_lines, lowered_twin, tree_rows
from nestforge.ir.names import ScopeChildren, normalize_labels
from nestforge.stages.moves import Rewrite, legal_moves, plan_move

#: NumPy dtype name -> ctypes scalar; DaCe lowers a comparison transient to C bool.
CTYPE = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "bool": ctypes.c_bool,
}


def corpus_kernel(name: str) -> CorpusKernel:
    """The corpus kernel ``name`` (registry key or manifest stem) names."""
    kernel = bench.corpus_kernel(name)
    assert kernel is not None, f"{name} is no corpus kernel with a DaCe program; the corpus this test pins changed"
    return kernel


def run(sdfg, inputs: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Run ``sdfg`` on copies of ``inputs`` with ``N=n``; returns the buffers after the call."""
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def random_vectors(n: int = 48, names: tuple[str, ...] = ("a", "b", "c"), seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


def signature_order(text: str, symbol: str, lang: str = "c") -> list[str]:
    """Parameter names of a translated kernel's entry, in declaration order: the emitted C order (sorted arrays,
    then symbols) is not the manifest's ``input_args`` order, so arguments bind to this."""
    if lang == "fortran":
        m = re.search(rf"subroutine\s+{re.escape(symbol)}\s*\((.*?)\)", text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise LookupError(f"subroutine {symbol} not found")
        return [a.strip() for a in m.group(1).replace("&", " ").split(",") if a.strip()]
    params = raw_signature(text, symbol)
    return [p.strip().split()[-1].lstrip("*") for p in params.split(",") if p.strip() and p.strip() != "void"]


def scalar_ctype(sdfg: dace.SDFG, name: str) -> type[ctypes._SimpleCData]:
    """ctype of a by-value argument: a float is ``c_double``, any integer ``int64_t`` whatever its SDFG width,
    since a narrower c_int leaves the upper register half undefined."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def c_argtypes(order: list[str], boundary: Boundary) -> list[CType]:
    """ctypes type per C parameter: an array is a pointer to its dtype, a float symbol a double, any other int64."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else scalar_ctype(sdfg, a)
        for a in order
    ]


def sdfg_to_python(sdfg: dace.SDFG, fn_name: str = "kernel") -> tuple[str, dace.SDFG]:
    """Standalone Python source for a whole SDFG, whose non-array arguments are its symbols, and the lowered copy it
    was emitted from (its descriptors size the scratch buffers the caller allocates). The symbols are every free
    symbol: ``arglist`` alone drops one only a library node's memlet step reads, which its expansion then indexes by."""
    arglist = sdfg.arglist(free_symbols=sdfg.free_symbols)
    symbols = [a for a in arglist if a not in sdfg.arrays]
    lowered = copy.deepcopy(sdfg)
    lower(lowered)
    widen_scratch(lowered, symbols)
    data_args = [a for a in arglist if a in sdfg.arrays]
    args = data_args + [s for s in scratch_arrays(lowered) if s not in data_args] + symbols
    return render(fn_name, args, lowered), lowered


def run_emitted(source: str, fn_name: str, lowered: dace.SDFG, inputs: dict, sizes: dict[str, int]) -> dict:
    """Call an emitted kernel C-style: every buffer its signature names is allocated here, from ``inputs`` (cast and
    reshaped to the descriptor; a scalar is a 1-element buffer) or zeroed; returns the buffers after the call."""
    kernel = vars(load_emitted(source, fn_name))[fn_name]
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    call = {}
    for name in inspect.signature(kernel).parameters:
        if name in sizes or name not in lowered.arrays:  # a symbol, by value
            call[name] = sizes[name] if name in sizes else inputs[name]
            continue
        desc = lowered.arrays[name]
        shape = tuple(int(symbolic.evaluate(d, env)) for d in desc.shape)
        dtype = np.dtype(desc.dtype.type)
        given = inputs.get(name)
        call[name] = np.zeros(shape, dtype) if given is None else np.array(given, dtype).reshape(shape)
    kernel(**call)
    return call


def fusion_moves(sdfg: dace.SDFG) -> list[tuple[str, tuple[str, ...]]]:
    """Every legal loop or map fusion of ``sdfg`` as ``(kind, labels)``, after making its tree labels unique."""
    normalize_labels(sdfg)
    return legal_moves(sdfg, "loop-fusion") + legal_moves(sdfg, "map-fusion")


def apply_move(sdfg: dace.SDFG, move: tuple[str, tuple[str, ...]]) -> str:
    """Apply one listed ``(kind, labels)`` move; returns the transformation that ran."""
    kind, labels = move
    rows = tree_rows(sdfg)
    plan = plan_move(kind, [rows[label] for label in labels])
    assert isinstance(plan, Rewrite), plan
    plan.commit()
    return plan.name


def fission_to_fixpoint(sdfg: dace.SDFG) -> int:
    """Apply the first legal loop or map fission until none remains; returns how many applied."""
    for applied in range(200):  # bound: each fission strictly adds a nest
        normalize_labels(sdfg)
        moves = legal_moves(sdfg, "loop-fission") + legal_moves(sdfg, "map-fission")
        if not moves:
            return applied
        apply_move(sdfg, moves[0])
    raise AssertionError("fission did not converge")


def extern_declaration_and_call(ext: ExternalCall, state: dace.SDFGState) -> tuple[str, str]:
    """The forward declaration and the call DaCe's ``ExternCall`` expansion emits for ``ext``."""
    tasklet = ExpandExternCall.expansion(ext, state, state.sdfg)
    return tasklet.code_global.as_string, tasklet.code.as_string


def compile_frame(frame: Path, folder: Path, name: str, opts: BuildOptions) -> tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so``, directly or, with ``link_external``, through an archive."""
    so = folder / f"lib{name}.so"
    if opts.link_external:
        return so, build_archive([frame], folder, folder / f"lib{name}_nest.a", so, opts)
    cmds = build_commands(folder, opts)
    obj = folder / f"{name}.o"
    t0 = time.perf_counter()
    toolchain.run([cmds.compiler, *cmds.cflags, "-c", *cmds.compile_extra, str(frame), "-o", str(obj)])
    link_shared(cmds.compiler, [*cmds.cflags, str(obj)], cmds.link_libs, so)
    return so, time.perf_counter() - t0


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, opts: BuildOptions | None = None) -> BuiltSDFG:
    """Generate, compile and link ``sdfg``; an OpenMP runtime is linked unless none is usable."""
    gen = generate_program(sdfg, out_dir)
    init_params = parse_params(signature(gen.source, f"__dace_init_{gen.name}"))
    prog_params = parse_params(signature(gen.source, f"__program_{gen.name}"))
    so, compile_seconds = compile_frame(gen.frame, gen.folder, gen.name, opts or BuildOptions())
    return BuiltSDFG(
        name=gen.name,
        so_path=so,
        lib=ctypes.CDLL(str(so)),
        init_params=init_params,
        prog_params=prog_params,
        codegen_seconds=gen.codegen_seconds,
        compile_seconds=compile_seconds,
    )


def emit_sources(prep: BenchKernel, out_dir: str | Path, target: str = "c") -> list[Path]:
    """Translate the kernel's Python source to ``target`` in-process with HPCAgent-Bench's translators.

    :returns: the generated source files, C then C++ then Fortran.
    """
    from hpcagent_bench import emit_bridge

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if emit_bridge.emit_kernel(prep.spec, prep.numpy_path, out, target=target, precision="float64") != 0:
        raise RuntimeError(f"numpyto refused {prep.name} (target={target}); the reason is on stderr")
    name = prep.name
    return sorted(out.glob(f"{name}_*.c")) + sorted(out.glob(f"{name}_*.cpp")) + sorted(out.glob(f"{name}_*.f90"))


def nest_to_python(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone Python source ``def <fn_name>(<args>): ...`` of an extracted nest."""
    sdfg = oracle_sdfg(boundary)
    return render(fn_name, kernel_args(boundary, kernel_arrays(boundary, sdfg)), sdfg)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: str | None = None) -> Boundary:
    """Outline a map nest or a control-flow nest; see :func:`extract_map_nest` and :func:`extract_cfg_nest`."""
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, CfgNest):
        return extract_cfg_nest(parent_sdfg, node, name=name)
    raise TypeError(
        f"cannot extract node of type {type(node).__name__}; expected MapEntry, LoopRegion, or ConditionalBlock"
    )


def kernel_body(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry, children: ScopeChildren) -> list[str]:
    """The Python statements a leaf kernel computes. ``children`` is the state's ``scope_children()``."""
    if any(isinstance(node, nodes.MapEntry) for node in children[entry]):
        return []
    return body_lines(lowered_twin(sdfg), state, entry)


#: Size of an integer symbol the caller gave no value for.
DEFAULT_SIZE = 1 << 16


@dataclass(frozen=True, slots=True)
class BenchKernel:
    """An extracted kernel written as the files HPCAgent-Bench's translator reads, with its ``BenchSpec``."""

    name: str
    numpy_path: Path
    yaml_path: Path
    numpy_source: str
    manifest: dict[str, Any]
    spec: Any


def bench_manifest(boundary: Boundary, name: str, sizes: dict[str, int] | None = None) -> dict[str, Any]:
    """The bench manifest of one extracted kernel, its signature the oracle's."""
    sdfg = oracle_sdfg(boundary)
    arrays = kernel_arrays(boundary, sdfg)
    shapes: dict[str, dict[str, str]] = {}
    for a in arrays:
        dims = [symbolic.symstr(d, cpp_mode=False) for d in sdfg.arrays[a].shape]
        shape = "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"
        shapes[a] = {"shape": shape, "dtype": np.dtype(sdfg.arrays[a].dtype.type).name}
    int_params: dict[str, int] = {}
    float_scalars: dict[str, float] = {}
    for s in boundary.symbols:
        # a float symbol is a staged value; the translator declares it double, not int64
        if s in sdfg.symbols and np.dtype(sdfg.symbols[s].type).kind == "f":
            float_scalars[s] = 0.0
        else:
            int_params[s] = int((sizes or {}).get(s, DEFAULT_SIZE))
    init: dict[str, Any] = {"arrays": shapes} | ({"scalars": float_scalars} if float_scalars else {})
    return {
        "name": name,
        "func_name": name,
        "relative_path": "extended",
        "level": 1,
        "parameters": {"S": int_params},
        "input_args": kernel_args(boundary, arrays),
        "array_args": arrays,
        "output_args": list(boundary.outputs),
        "init": init,
    }


def bench_kernel(boundary: Boundary, name: str, out_dir: Path, sizes: dict[str, int] | None = None) -> BenchKernel:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` for one extracted kernel and build its ``BenchSpec``."""
    from hpcagent_bench.spec import BenchSpec

    out_dir.mkdir(parents=True, exist_ok=True)
    source = prepare(boundary, name).numpy_source
    manifest = bench_manifest(boundary, name, sizes)
    numpy_path = out_dir / f"{name}_numpy.py"
    numpy_path.write_text(source)
    yaml_path = out_dir / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    spec = BenchSpec.from_dict(manifest | {"short_name": name, "module_name": name}, source=str(yaml_path))
    return BenchKernel(name, numpy_path, yaml_path, source, manifest, spec)
