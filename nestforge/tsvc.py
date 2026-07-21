"""TSVC corpus adapter: the SDFG source (DaCe ``@dace.program``) and the native-baseline source
(OptArena ``foundation`` track) for the TSVC-2 microkernels.

Two upstreams, joined per kernel by its short name (``s000``, ``vpv``, ...):

  * **SDFG source** -- DaCe's ``performance_regression_jobs/tsvc_corpus.py`` holds the 151 curated,
    typed ``@dace.program`` kernels. OptArena's ``foundation`` track ships each kernel's oracle +
    manifest + original C but no ``_dace.py``, so the SDFG comes from here -- loaded by file path
    (a script, not a package), overridable with ``NESTFORGE_TSVC_CORPUS``.
  * **native baseline** -- ``foundation/tsvc_2_<key>_original.cpp`` is the original scalar TSVC loop
    as a bare ``extern "C"`` kernel, symbol ``<key>_d``: the "how well does this compiler
    auto-vectorize the reference loop" column of the arena.

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
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace import symbolic
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.symbol_propagation import SymbolPropagation

from optarena.initialize import fill_index_array
from optarena.spec import KERNELS, BenchSpec

from nestforge.arena import resolve_shape
from nestforge.extract import trip_count_symbols
from nestforge.fusion import get_fusion_strategy

#: Shape symbols the sizing logic samples/fixes; every other boundary symbol is a scalar loop
#: parameter (taken from the kernel's registered ``params``) or the corpus multiplier ``S``.
_SHAPE_SYMS = ("LEN_1D", "LEN_2D", "LEN_3D")
#: fixed preset scale per shape symbol (mirrors the OptArena presets; ``XL`` ``LEN_1D`` is ~2 GiB fp64),
#: used by the cross-language XL job so every compiler/language sees the same size for a preset.
#: ``PROF`` is the full-matrix job's PROFILING size: chosen so ONE fp64 array clearly exceeds the
#: GH200 Grace L3 (~114 MB/socket) -- each dim gives ~128 MiB/array -- landing timings in the realistic
#: memory-bound regime rather than in-cache. Smaller than XL (dominated by alloc/first-touch, too slow
#: for a huge sweep) but still out of L3. See perf/README_tsvc_full.md for the rationale.
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
    beside each entry of ``dace.__path__`` (package dir or repo root depending on the editable-install
    layout, so both it and its parent are tried)."""
    override = os.environ.get("NESTFORGE_TSVC_CORPUS") if filename == _CORPUS_FILE["tsvc2"] else None
    cands = [Path(override)] if override else []
    for p in dace.__path__:
        base = Path(p)
        cands.append(base / "performance_regression_jobs" / filename)
        cands.append(base.parent / "performance_regression_jobs" / filename)
    return cands


@functools.lru_cache(maxsize=None)
def corpus_module(corpus: str = "tsvc2") -> ModuleType:
    """Load a DaCe TSVC corpus script by path (a script, not an importable package). ``tsvc2`` =
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


@functools.lru_cache(maxsize=None)
def corpus_symbol_values(corpus: str = "tsvc2") -> Dict[str, int]:
    """Concrete values for a corpus' own SCALAR symbols -- multiplier ``S`` plus tile/offset symbols like
    ``T``/``K``/``SSYM`` -- as the corpus script itself declares them.

    Both spellings are read rather than either assumed: ``tsvc2`` binds a lone ``S_VALUE``, ``tsvc2_5`` a
    ``SIZES`` dict (``S`` differs between them), so a ``tsvc2_5`` kernel resolves its symbols instead of
    raising on a missing ``S_VALUE``. The value itself is irrelevant to a reference-vs-candidate
    comparison (both sides see the same one); it must just be concrete and the corpus' own, since loop
    bounds are written to stay in range for it.

    ``_SHAPE_SYMS`` are dropped: sizing those is the arena's job (the preset/random sweep IS the size
    axis), so a corpus' ``LEN_1D`` must never shadow the sampled one. Read via ``vars`` since a corpus
    script is a module namespace, not an object with an API."""
    namespace = vars(corpus_module(corpus))
    values = dict(namespace.get("SIZES", {}))
    if "S_VALUE" in namespace:
        values.setdefault("S", namespace["S_VALUE"])
    return {sym: int(val) for sym, val in values.items() if sym not in _SHAPE_SYMS}


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
    def foundation_entry(self) -> Optional[Path]:
        """This kernel's OptArena manifest path from the registry, or ``None`` with no foundation entry.

        Each kernel keeps its manifest, native baseline and numpy together in a ``foundation/<stem>/``
        subfolder -- ``tsvc2`` under the ``tsvc_2_<key>`` stem, ``tsvc2_5`` under the bare ``<key>`` -- so
        both spellings are tried (prefixed first) and a ``tsvc2_5`` kernel reaches its own subfolder
        instead of silently falling back to defaults."""
        for stem in (f"tsvc_2_{self.key}", self.key):
            registry_key = f"foundation/{stem}/{stem}"
            if registry_key in KERNELS:
                return Path(KERNELS[registry_key])
        return None

    @property
    def native_cpp(self) -> Optional[Path]:
        """The ``_original.cpp`` native-baseline source, or ``None`` if this kernel has no foundation entry."""
        entry = self.foundation_entry
        if entry is None:
            return None
        cpp = entry.with_name(f"{entry.stem}_original.cpp")
        return cpp if cpp.exists() else None

    @property
    def native_symbol(self) -> str:
        return f"{self.key}_d"

    @property
    def yaml_path(self) -> Optional[Path]:
        """The kernel's OptArena manifest, or ``None`` when it has no foundation entry."""
        entry = self.foundation_entry
        return entry if entry is not None and entry.exists() else None

    @property
    def optarena_name(self) -> Optional[str]:
        """The kernel's OptArena short name (the manifest stem), or ``None`` when it has no manifest."""
        return self.yaml_path.stem if self.yaml_path is not None else None


def iter_tsvc_kernels(only: Optional[List[str]] = None, corpus: str = "tsvc2") -> List[TsvcKernel]:
    """Every kernel from a DaCe TSVC corpus, optionally filtered to the ``only`` short names.

    ``tsvc2`` ships a ``KERNELS`` registry of descriptors (``s000_d_single`` -> key ``s000``, with
    regime/params/tags); ``tsvc2_5`` ships a ``collect()`` of bare ``@dace.program`` s (key = function
    name, no regime/params metadata)."""
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

    Pre-split axis:
    - ``"simplify-parallel"`` -- ``simplify`` + ``LoopToMap`` + ``MapFusion`` (V+H) + ``simplify``: the
      reference map-nest form the external compiler vectorizes/tiles.
    - ``"canonicalize"``      -- DaCe's extended-branch statement-level normal form (normalized
      loops/maps, lifted reductions, parallelized where sound).
    - ``"auto-opt"``          -- DaCe's full CPU ``auto_optimize`` pipeline, measured as its own axis
      rather than assumed.

    ``SymbolPropagation`` runs for EVERY mode before the branch: the frontend names a derived loop bound
    with a fresh symbol on an inter-state edge (s122's ``n1_minus_1 = n1 - 1``), which stays outside the
    nest when the splitter peels the loop, leaving the DERIVED name as an unbindable free argument.
    Propagating first rewrites the bound so the real parameter (``n1``) reappears. ``canonicalize``
    already runs the pass internally (idempotent, so re-running costs nothing); ``simplify-parallel`` and
    ``auto-opt`` didn't, silently dropping s122/s172/s4114 since DaCe's ``Simplify`` doesn't include it.
    """
    sdfg = kernel.program.to_sdfg(simplify=True)
    SymbolPropagation().apply_pass(sdfg, {})
    if opt_mode == "simplify-parallel":
        get_fusion_strategy("maximal-fusion")(sdfg)  # Phase 1: fuse everything legal
    elif opt_mode == "canonicalize":
        canonicalize(sdfg, target="cpu")
    elif opt_mode == "auto-opt":
        auto_optimize(sdfg, dace.DeviceType.CPU)
    else:
        raise ValueError(f"unknown opt_mode {opt_mode!r}; expected one of {OPT_MODES}")
    # Forced for EVERY mode: only `canonicalize` runs it internally, and a residual sympy floor()
    # reaches codegen as an index that truncates term by term.
    NormalizeFloorDivision().apply_pass(sdfg, {})
    return sdfg


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

    A symbol that sizes an array (``LEN_1D``/``LEN_2D``/``LEN_3D``) takes the fixed ``preset`` scale
    (``S``/``M``/``L``/``XL``) when given -- so every compiler/language sees the same size -- else the
    OptArena preset range (random when ``random_sizes``, seeded; else the ``M`` default). A registered
    scalar parameter takes its own value; a corpus-bound scalar (``S``, ``T``, ``K``, ...) takes
    :func:`corpus_symbol_values`.

    A leftover symbol takes ``0`` in exactly two proven-safe cases:

    * the standalone SDFG never takes it as an argument -- unreachable, so 0 is as good as anything;
    * it can't change the nest's TRIP COUNT (absent from :func:`trip_count_symbols` and every array
      shape) -- it only picks WHICH element is touched, never HOW MUCH work runs, so oracle and
      candidate still compare real, full-size work. Covers a leaked OUTER INDEX the nest only reads
      (s1115: ``i`` in ``aa[i, j]``, no bound) and a leaked INDUCTION START it reads+increments
      (s123/s124/s125/... ``j``/``k``): the corpus inits the counter before the loop and the splitter
      leaves that init outside the nest, so it enters as an unbindable free argument.

      This does NOT claim 0 is the corpus's real starting value (s123 shifts to ``a[0..]`` instead of
      ``a[-1..]``) -- the lowering is validated, not the kernel's exact initial state. Resolving the
      true value from ``boundary.parent_sdfg`` is the proper follow-up.

    Anything else is a symbol the nest reads that CAN change its trip count and that nothing here can
    value -- raised on, since sizing it 0 would validate vacuously against a degenerate bound. A loud
    skip beats a silent green.
    """
    rng = np.random.default_rng(seed + key_seed(kernel.key))
    presets = yaml_presets(kernel)
    shape_syms = shape_symbols(boundary.standalone_sdfg)
    corpus_values = corpus_symbol_values(kernel.corpus)
    # The arguments the nest actually takes, and the symbols that can change how much work it does:
    # together these say whether a leftover symbol's value can reach -- or degenerate -- the computation
    # (see the 0-vs-raise below). Shapes join the trip-count set: both size the work.
    nest_arglist = set(boundary.standalone_sdfg.arglist())
    work_syms = trip_count_symbols(boundary.standalone_sdfg) | shape_syms
    sizes: Dict[str, int] = {}
    for sym in boundary.symbols:
        if sym in kernel.params:
            sizes[sym] = int(kernel.params[sym])
        elif sym in corpus_values and sym not in shape_syms:
            # Guarded on shape_syms: a corpus scalar that also sizes a buffer here is the sampler's to size.
            sizes[sym] = corpus_values[sym]
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
        elif sym not in nest_arglist:
            sizes[sym] = 0  # never passed to the nest: the value provably cannot reach the computation
        elif sym not in work_syms:
            # A leaked index -- consumed (s1115's outer `i`) or self-incremented (s123's `j`). It selects
            # WHICH element the nest touches, never HOW MUCH work it does, so 0 keeps the iteration space
            # and the buffers full-size and both sides see the same one. See the docstring.
            sizes[sym] = 0
        else:
            raise ValueError(f"{kernel.corpus} kernel {kernel.key!r}: boundary symbol {sym!r} is read by the nest and "
                             f"sizes its work (it appears in a loop bound, a map range, an inter-state condition or an "
                             f"array shape), but is not a registered parameter, a corpus symbol "
                             f"({sorted(corpus_values)}) or a known shape symbol, so nothing here knows its value. "
                             f"Sizing it 0 would validate vacuously against a degenerate result -- bind it in the "
                             f"corpus (SIZES/S_VALUE) or in the kernel's params.")
    return sizes


def index_fills(kernel: TsvcKernel, boundary, sizes: Dict[str, int], seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """Valid-subscript values for the nest's integer INDEX arrays, as the kernel's OptArena manifest
    declares them. Feed the result to :func:`nestforge.arena.make_inputs` as ``given``.

    The manifest, not a nest-forge guess, says what belongs in a benchmark's own arrays. The case that
    bites is the index array: the manifest declares e.g. ``ip: int32`` filled with a PERMUTATION of
    ``[0, N)``, whereas the default uniform-float fill cast to int collapses to ALL-ZEROS -- degrading a
    gather ``a[i] = b[ip[i]]`` to a cached read of ``b[0]`` and turning OptArena's conflict-FREE scatter
    permutation into a race on ``a[0]`` once lowered to a ``dace.map``.

    Only arrays the MANIFEST declares with an integer dtype are filled (an int array isn't automatically
    a subscript -- the manifest separates index ``ip`` from mask), and only ones the nest actually READS
    (an unused ``(LEN_2D,LEN_2D)`` fp64 buffer is 2 GiB at ``XL``). The fill takes the SDFG descriptor's
    dtype, not the manifest's, since that's the width the compiled code reads across the ABI.

    ``seed=None`` draws fresh entropy (fuzz); an int pins the fill. Returns ``{}`` for a kernel with no
    manifest or whose manifest declares no index array.
    """
    return index_fills_for_manifest(kernel.optarena_name, boundary, sizes, seed=seed)


def index_fills_for_manifest(manifest_name: Optional[str],
                             boundary,
                             sizes: Dict[str, int],
                             seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """:func:`index_fills` keyed by the OptArena manifest NAME rather than a :class:`TsvcKernel`.

    Fills depend only on the manifest and the nest, not on anything TSVC-specific, so a caller holding a
    bare manifest name (iterating OptArena kernels rather than the TSVC corpus) gets the same
    valid-subscript fills instead of the all-zeros default. ``None`` -> ``{}``.
    """
    if manifest_name is None:
        return {}
    spec = BenchSpec.load(manifest_name)
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
