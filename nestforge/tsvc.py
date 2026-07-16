"""TSVC corpus adapter: the SDFG source (DaCe ``@dace.program``) and the native-baseline source
(OptArena ``foundation`` track) for the TSVC-2 microkernels.

Two upstreams, joined per kernel by its short name (``s000``, ``vpv``, ...):

  * **SDFG source** -- DaCe's ``performance_regression_jobs/tsvc_corpus.py`` holds the 151 curated,
    typed ``@dace.program`` kernels (``s000_d_single``). OptArena's ``foundation`` track ships each
    kernel's numpy oracle + manifest + original C but **no** ``_dace.py``, so the SDFG must come from
    here. It is loaded by file path (it is a script, not an importable package) located from the
    installed DaCe -- overridable with ``NESTFORGE_TSVC_CORPUS`` for a non-standard layout.
  * **native baseline** -- OptArena ``foundation/tsvc_2_<key>_original.cpp`` is the original scalar
    TSVC loop as a bare ``extern "C"`` kernel (timing instrumentation stripped), symbol ``<key>_d``.
    It is the "how well does this compiler auto-vectorize the reference loop" column of the arena.

The two need not be in bijection: a kernel with an SDFG but no ``_original.cpp`` still runs the
extracted-nest columns (no native column); the intersection is what the plan calls "the 135".
"""
from __future__ import annotations

import functools
import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import dace
from dace import symbolic
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize import canonicalize

from optarena.initialize import fill_index_array
from optarena.spec import KERNELS, BenchSpec

from nestforge.arena import resolve_shape

#: Shape symbols the sizing logic samples/fixes; every other boundary symbol is a scalar loop
#: parameter (taken from the kernel's registered ``params``) or the corpus multiplier ``S``.
_SHAPE_SYMS = ("LEN_1D", "LEN_2D", "LEN_3D")
#: fixed preset scale per shape symbol (mirrors the OptArena presets; ``XL`` ``LEN_1D`` is ~2 GiB fp64).
#: Used by the cross-language XL job so every compiler/language sees the same size for a preset.
#: ``PROF`` is the full-matrix job's PROFILING size: chosen so ONE fp64 array clearly exceeds the
#: GH200 Grace L3 (~114 MB/socket) -- LEN_1D=2**24 -> 128 MiB/array, LEN_2D=4096 -> 4096**2*8=128 MiB,
#: LEN_3D=256 -> 256**3*8=128 MiB -- so timings land in the realistic memory-bound regime rather than an
#: in-cache best case (as most kernels use >=2 arrays, the working set is a multiple of that). It is
#: SMALLER than XL (which is dominated by allocation/first-touch time, too slow for a huge sweep) but
#: still out of L3. See docs / perf/README_tsvc_full.md for the rationale.
_PRESET = {
    "LEN_1D": {
        "S": 512,
        "M": 32768,
        "L": 131072,
        "PROF": 16777216,
        "XL": 268435455
    },
    "LEN_2D": {
        "S": 64,
        "M": 256,
        "L": 512,
        "PROF": 4096,
        "XL": 1024
    },
    "LEN_3D": {
        "S": 16,
        "M": 48,
        "L": 96,
        "PROF": 256,
        "XL": 200
    },
}
#: Fallback ``(lo, hi)`` sample range per shape symbol when the OptArena yaml carries no preset for
#: it (a 1D kernel's yaml lists only ``LEN_1D``, so a 2D scratch's ``LEN_2D`` falls back here). The
#: ``LEN_2D`` ceiling keeps the square ``LEN_2D**2`` buffers to a few MB.
_SYM_RANGE = {"LEN_1D": (4096, 131072), "LEN_2D": (64, 512)}
#: Fixed size per shape symbol when ``--random-sizes`` is off (the OptArena ``M`` preset for
#: ``LEN_1D``; a moderate square for ``LEN_2D``).
_SYM_FIXED = {"LEN_1D": 32768, "LEN_2D": 256}
#: Preset names never sampled: OptArena's ``XL`` ``LEN_1D`` is 268435455 (~2 GiB fp64), impractical.
_SKIP_PRESETS = frozenset({"XL"})

#: corpus name -> the DaCe ``performance_regression_jobs`` script that defines its ``@dace.program`` s.
_CORPUS_FILE = {"tsvc2": "tsvc_corpus.py", "tsvc2_5": "tsvc_2_5_corpus.py"}


def corpus_candidates(filename: str) -> List[Path]:
    """Where a corpus script might live: an explicit override, then ``performance_regression_jobs``
    beside each entry of ``dace.__path__`` (which is the package dir or the repo root depending on the
    editable-install layout, so both it and its parent are tried)."""
    override = os.environ.get("NESTFORGE_TSVC_CORPUS") if filename == _CORPUS_FILE["tsvc2"] else None
    cands = [Path(override)] if override else []
    for p in dace.__path__:
        base = Path(p)
        cands.append(base / "performance_regression_jobs" / filename)
        cands.append(base.parent / "performance_regression_jobs" / filename)
    return cands


@functools.lru_cache(maxsize=None)
def corpus_module(corpus: str = "tsvc2") -> ModuleType:
    """Load a DaCe TSVC corpus script by path (they are scripts, not importable packages). ``tsvc2`` =
    ``tsvc_corpus.py`` (151 kernels), ``tsvc2_5`` = ``tsvc_2_5_corpus.py`` (65 kernels)."""
    if corpus not in _CORPUS_FILE:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(_CORPUS_FILE)}")
    filename = _CORPUS_FILE[corpus]
    src = next((c for c in corpus_candidates(filename) if c.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"cannot locate DaCe's performance_regression_jobs/{filename} (the {corpus} @dace.program "
            f"source); looked in {[str(c) for c in corpus_candidates(filename)]}. Set "
            "NESTFORGE_TSVC_CORPUS to its path.")
    spec = importlib.util.spec_from_file_location(f"nestforge_{corpus}_corpus", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def foundation_dir() -> Path:
    """The OptArena ``foundation`` benchmark directory (holds the ``tsvc_2_*_original.cpp`` baselines)."""
    for short_name in KERNELS:  # KernelRegistry: iterate keys, index for the yaml path (not a dict -> no .items())
        if short_name.startswith("foundation/tsvc_2_"):
            return Path(KERNELS[short_name]).parent
    raise FileNotFoundError("no foundation/tsvc_2_* kernel found in the OptArena registry")


@dataclass
class TsvcKernel:
    """One TSVC kernel: its DaCe ``@dace.program`` SDFG source + its OptArena native baseline."""
    key: str  # short name, e.g. "s000" (the corpus "s000_d_single" with the "_d_single" suffix dropped)
    program: object  # the DaCe ``@dace.program``
    regime: str  # "1d" or "2d"
    params: Dict[str, int]  # scalar loop parameters (e.g. n1, n3) with their registered values
    corpus: str = "tsvc2"  # which corpus this kernel came from ("tsvc2" | "tsvc2_5")
    tags: frozenset = field(default_factory=frozenset)

    @property
    def native_cpp(self) -> Optional[Path]:
        """The ``_original.cpp`` native-baseline source, or ``None`` if this kernel has no foundation entry."""
        p = foundation_dir() / f"tsvc_2_{self.key}_original.cpp"
        return p if p.exists() else None

    @property
    def native_symbol(self) -> str:
        return f"{self.key}_d"

    @property
    def yaml_path(self) -> Optional[Path]:
        """The kernel's OptArena manifest, or ``None`` when it has no foundation entry.

        The two corpora name their manifests differently: a ``tsvc2`` kernel is ``tsvc_2_<key>.yaml``
        while a ``tsvc2_5`` kernel is a bare ``<key>.yaml`` (``reroll_gather.yaml``). Both are checked --
        the prefixed name first -- so a ``tsvc2_5`` kernel reaches its presets and its declared index
        arrays instead of silently falling back to the defaults. The two namespaces do not overlap.
        """
        for filename in (f"tsvc_2_{self.key}.yaml", f"{self.key}.yaml"):
            p = foundation_dir() / filename
            if p.exists():
                return p
        return None

    @property
    def optarena_name(self) -> Optional[str]:
        """The kernel's OptArena short name (the manifest stem), or ``None`` when it has no manifest."""
        return self.yaml_path.stem if self.yaml_path is not None else None


def iter_tsvc_kernels(only: Optional[List[str]] = None, corpus: str = "tsvc2") -> List[TsvcKernel]:
    """Every kernel from a DaCe TSVC corpus, optionally filtered to the ``only`` short names.

    ``tsvc2`` ships a ``KERNELS`` registry of ``TSVCKernel`` descriptors (name ``s000_d_single`` -> key
    ``s000``, with regime/params/tags); ``tsvc2_5`` ships a ``collect()`` of bare ``@dace.program`` s
    (key = the program's function name, no regime/params metadata)."""
    module = corpus_module(corpus)
    want = set(only) if only else None
    out: List[TsvcKernel] = []
    if corpus == "tsvc2_5":
        for prog in module.collect():
            key = prog.f.__name__
            if want is not None and key not in want:
                continue
            out.append(TsvcKernel(key=key, program=prog, regime="1d", params={}, corpus=corpus))
        return out
    for k in module.KERNELS:
        key = k.name[:-len("_d_single")] if k.name.endswith("_d_single") else k.name
        if want is not None and key not in want:
            continue
        out.append(
            TsvcKernel(key=key, program=k.program, regime=k.regime, params=dict(k.params), corpus=corpus, tags=k.tags))
    return out


#: The optimization modes applied to a kernel's SDFG *before* the loopnest-splitting pass runs. This is
#: the DaCe-side opt-pipeline axis: each value is a distinct lowering the external compiler then sees.
OPT_MODES = ("simplify-parallel", "canonicalize", "auto-opt")


def build_sdfg(kernel: TsvcKernel, opt_mode: str = "simplify-parallel") -> dace.SDFG:
    """The kernel's SDFG after an optimization mode, ready for the loopnest-splitting pass.

    Optimization modes (the pre-split axis):
    - ``"simplify-parallel"`` -- DaCe ``simplify`` + ``LoopToMap`` + ``MapFusion`` (V+H) + ``simplify``;
      the reference map-nest form the external compiler vectorizes/tiles. (Formerly ``"baseline"``.)
    - ``"canonicalize"``      -- the DaCe extended-branch canonicalization pipeline (statement-level
      normal form: normalized loops/maps, lifted reductions, parallelized where sound).
    - ``"auto-opt"``          -- DaCe's full CPU ``auto_optimize`` pipeline (greedy map fusion, tiling,
      transient reuse, library-node specialization) as DaCe would ship it -- the strongest DaCe-side
      lowering, measured as its own axis rather than assumed.
    """
    sdfg = kernel.program.to_sdfg(simplify=True)
    if opt_mode == "simplify-parallel":
        sdfg.apply_transformations_repeated([LoopToMap])
        sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal])
        sdfg.simplify()
        return sdfg
    if opt_mode == "canonicalize":
        canonicalize(sdfg, target="cpu")
        return sdfg
    if opt_mode == "auto-opt":
        auto_optimize(sdfg, dace.DeviceType.CPU)
        return sdfg
    raise ValueError(f"unknown opt_mode {opt_mode!r}; expected one of {OPT_MODES}")


def yaml_presets(kernel: TsvcKernel) -> Dict[str, List[int]]:
    """``{shape_symbol: [preset sizes]}`` from the kernel's OptArena yaml, skipping the ``XL`` preset."""
    if kernel.yaml_path is None:
        return {}
    params = yaml.safe_load(kernel.yaml_path.read_text()).get("parameters", {})
    presets: Dict[str, List[int]] = {}
    for name, mapping in params.items():
        if name in _SKIP_PRESETS or not isinstance(mapping, dict):
            continue
        for sym, size in mapping.items():
            presets.setdefault(sym, []).append(int(size))
    return presets


def key_seed(key: str) -> int:
    """A stable per-kernel seed offset (so every kernel gets different random sizes, reproducibly).
    ``hash`` is salted per process, so derive from the characters instead."""
    return sum((i + 1) * ord(c) for i, c in enumerate(key)) & 0xFFFF


def shape_symbols(sdfg: dace.SDFG) -> set:
    """Symbols that appear in an array's shape (the ones that actually size a buffer)."""
    return {
        str(s)
        for desc in sdfg.arrays.values()
        for dim in desc.shape
        for s in symbolic.pystr_to_symbolic(dim).free_symbols
    }


def sample_sizes(kernel: TsvcKernel,
                 boundary,
                 seed: int = 0,
                 random_sizes: bool = False,
                 preset: Optional[str] = None) -> Dict[str, int]:
    """A concrete value for every boundary symbol.

    A symbol that sizes an array (``LEN_1D``/``LEN_2D``/``LEN_3D``) is set from the fixed ``preset``
    scale (``S``/``M``/``L``/``XL``) when ``preset`` is given -- so every compiler and language sees the
    same size -- else drawn from the OptArena preset range (randomly when ``random_sizes``, seeded;
    otherwise the ``M``-scale default). A registered scalar parameter takes its value, ``S`` the corpus
    ``S_VALUE``, and a symbol that sizes nothing (a loop-carried index leaked into the boundary, e.g.
    ``i``) takes ``0`` -- the nest resets it before use, keeping oracle and candidate in agreement.
    """
    rng = np.random.default_rng(seed + key_seed(kernel.key))
    presets = yaml_presets(kernel)
    shape_syms = shape_symbols(boundary.standalone_sdfg)
    sizes: Dict[str, int] = {}
    for sym in boundary.symbols:
        if sym in kernel.params:
            sizes[sym] = int(kernel.params[sym])
        elif sym == "S":
            sizes[sym] = int(corpus_module(kernel.corpus).S_VALUE)
        elif preset and sym in _PRESET:
            sizes[sym] = int(_PRESET[sym][preset])
        elif sym in _SHAPE_SYMS and sym in shape_syms and sym in _SYM_RANGE:
            lo, hi = _SYM_RANGE[sym]
            if presets.get(sym):
                lo, hi = min(presets[sym]), max(presets[sym])
            if sym == "LEN_2D":
                hi = min(hi, _SYM_RANGE["LEN_2D"][1])  # keep LEN_2D**2 buffers bounded regardless of preset
                lo = min(lo, hi)
            sizes[sym] = int(rng.integers(lo, hi + 1)) if random_sizes else min(_SYM_FIXED[sym], hi)
        elif sym in shape_syms:
            # a shape symbol outside the known 1D/2D pair with no preset (e.g. LEN_3D in the non-preset
            # path): a small default so a 3D buffer stays modest (LEN_3D is only used by the preset job).
            sizes[sym] = _PRESET.get(sym, {}).get("M", 64)
        else:
            sizes[sym] = 0  # a loop-carried index that sizes nothing
    return sizes


def index_fills(kernel: TsvcKernel, boundary, sizes: Dict[str, int], seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """Valid-subscript values for the nest's integer INDEX arrays, as the kernel's OptArena manifest
    declares them. Feed the result to :func:`nestforge.arena.make_inputs` as ``given``.

    A nest's NON-transient arrays are the benchmark's own arrays, so the manifest -- not a nest-forge
    guess -- says what belongs in them. The case that bites is the integer index array: the manifest
    declares e.g. ``ip: int32`` and fills it with a PERMUTATION of ``[0, N)``, whereas the default
    uniform float fill cast to an integer dtype collapses to ALL-ZEROS. That silently degrades a gather
    ``a[i] = b[ip[i]]`` into a single cached read of ``b[0]`` -- so the arena times the wrong memory
    behaviour -- and inverts a scatter ``a[ip[i]] = ...`` from OptArena's guaranteed conflict-FREE
    permutation into a maximal write conflict on ``a[0]``, which is an outright race once the nest
    lowers to a ``dace.map``.

    Only arrays the MANIFEST declares with an integer dtype are filled: an integer array is not
    automatically a subscript, and the manifest is what separates an index (``ip``) from a mask. Only
    arrays the nest actually READS are materialized -- the manifest declares every array of the whole
    kernel, and an unused ``(LEN_2D,LEN_2D)`` fp64 buffer is 2 GiB at the ``XL`` preset.

    The fill takes the SDFG descriptor's dtype, not the manifest's: this buffer crosses the ABI as the
    kernel's own argument, so its width must be the one the compiled code reads.

    ``seed=None`` draws fresh entropy (fuzz); an int pins the fill. Returns ``{}`` for a kernel with no
    manifest and for one whose manifest declares no index array.
    """
    if kernel.optarena_name is None:
        return {}
    spec = BenchSpec.load(kernel.optarena_name)
    if spec.init is None:
        return {}
    rng = np.random.default_rng(seed)
    arrays = boundary.standalone_sdfg.arrays
    fills: Dict[str, np.ndarray] = {}
    for name, declared in sorted(spec.init.dtypes.items()):
        if np.dtype(declared).kind not in "iu" or name not in boundary.inputs:
            continue
        dtype = np.dtype(arrays[name].dtype.type)
        if dtype.kind not in "iu":
            continue  # the manifest calls it an index but the nest holds it as a float: not a subscript
        fills[name] = fill_index_array(resolve_shape(arrays[name].shape, sizes), dtype, rng=rng)
    return fills


def native_signature(cpp_text: str, symbol: str) -> List[Tuple[str, str, bool]]:
    """Parse the ``extern "C"`` baseline signature ``void <symbol>(...)`` into
    ``[(param_name, base_ctype, is_pointer), ...]`` in declaration order.

    Base ctype is one of ``double``/``float``/``int64_t``/``int`` (the only types the TSVC baselines
    use); ``const`` / ``__restrict__`` qualifiers are stripped.
    """
    m = re.search(rf"\b{re.escape(symbol)}\s*\((.*?)\)", cpp_text, re.S)
    if not m:
        raise LookupError(f"native symbol {symbol} not found in the baseline source")
    params: List[Tuple[str, str, bool]] = []
    for raw in m.group(1).split(","):
        tok = raw.replace("__restrict__", " ").replace("const", " ").strip()
        if not tok:
            continue
        is_ptr = "*" in tok
        name = re.split(r"[\s*]+", tok)[-1]
        base = ("int64_t" if "int64" in tok else "double" if "double" in tok else "float" if "float" in tok else "int")
        params.append((name, base, is_ptr))
    return params
