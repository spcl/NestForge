"""One entry point over the 4-phase optimizer: take whatever the caller has, pick a search space,
measure the variants.

Language- and compiler-defined semantics are not trusted -- whether a vectorizer flag, an fp mode or
a codegen path is faster is decided by compiling and timing it. What may vary depends only on the
input:

=============================  =========================  ====================================
input                          what we can still change   search space
=============================  =========================  ====================================
C / C++ / Fortran SOURCE       only how it is compiled    vectorize x fp                    (9)
NumPy / Fortran / SDFG         the generated code too     + codegen knobs, budgeted        (72)
=============================  =========================  ====================================

per compiler; the arena builds each variant once per discovered toolchain.

An agent does not change the space, it CONTRIBUTES: an :class:`AgentVariant` carries finished source,
an exact flag set, or both, measured on the same footing as every enumerated variant. Fixing the
space keeps steered and unsteered runs over identical ground, so their difference measures the agent.

:func:`plan_search` is pure -- no compiler, no filesystem -- so the contract is testable on a machine
with no toolchain. This module PLANS only; nothing here runs the arena yet (see README, Known gaps).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import dace

#: Suffixes that name a source we can hand STRAIGHT to a compiler (case A).
COMPILABLE_SUFFIXES = {
    '.c': 'c_source',
    '.cc': 'cpp_source',
    '.cpp': 'cpp_source',
    '.cxx': 'cpp_source',
    '.f': 'fortran_source',
    '.f90': 'fortran_source',
    '.f03': 'fortran_source',
    '.f08': 'fortran_source',
}

#: Suffixes we PARSE into an SDFG (case B). Fortran also appears above -- see :func:`classify_input`.
PARSEABLE_SUFFIXES = {'.py': 'numpy', '.sdfg': 'sdfg', '.sdfgz': 'sdfg'}


class InputKind(enum.Enum):
    """What the caller handed us, after disambiguation."""
    # no __slots__: Enum
    C_SOURCE = 'c_source'
    CPP_SOURCE = 'cpp_source'
    FORTRAN_SOURCE = 'fortran_source'  # compile as-is
    FORTRAN_PARSE = 'fortran_parse'  # lower to an SDFG via dace-fortran
    NUMPY = 'numpy'
    SDFG = 'sdfg'


class SearchSpace(enum.Enum):
    """Which axes the sweep may move. The input decides; an agent never changes it."""
    # no __slots__: Enum
    FLAGS = 'flags'  # the code is fixed, only the compiler invocation varies
    CODEGEN = 'codegen'  # we generate the C++, so codegen varies too


#: Kinds that arrive as finished source. These can never reach the codegen axes.
PROVIDED_SOURCE_KINDS = frozenset({InputKind.C_SOURCE, InputKind.CPP_SOURCE, InputKind.FORTRAN_SOURCE})
#: Kinds we lower to an SDFG ourselves.
PARSED_KINDS = frozenset({InputKind.NUMPY, InputKind.FORTRAN_PARSE, InputKind.SDFG})

#: Compiler-invocation axes, resolved to per-compiler flags at execution time; named here so the
#: contract stays inspectable without a compiler.
#:
#: `vectorize` is ONE axis, not a switch crossed with a cost model: with vectorization off the cost
#: model has nothing to decide, so the crossed form emitted the same variant three times.
#: `fp` mirrors :data:`nestforge.arena.FP_MODES`, which owns the flag lists and the per-mode
#: comparison tolerance. Restated here to keep planning import-light; a test asserts they never drift.
FLAG_AXES: Dict[str, Sequence[str]] = {
    'vectorize': ('none', 'cheap', 'auto'),
    'fp': ('ieee-strict', 'fast-but-ieee', 'fast-math'),
}

#: Every DaCe CPU codegen axis, values verified against dace `extended` config_schema.yml.
#: `implementation` is the old-vs-new switch; the rest are readable-generator knobs, no-ops under
#: `legacy`.
CODEGEN_AXES: Dict[str, Sequence] = {
    'implementation': ('legacy', 'experimental_readable'),
    'const_scalar_abi': ('by_ref', 'by_value'),
    'index_ctype': ('int64_t', 'int32_t'),
    'heap_ptr_restrict': ('restrict', 'may_alias', 'none'),
    'index_fn_qualifier': ('inline_constexpr', 'inline', 'always_inline'),
    'loop_index_type': ('auto', 'int32_t', 'int64_t'),
    'loop_bound_cmp': ('lt', 'le', 'ne'),
    'inline_full_array_nsdfg': (False, True),
    'split_nsdfg_translation_units': (False, True),
    'external_translation_units': (False, True),
    'scalar_emission_type': ('scalar', 'keep'),
    'explicit_copy': ('on', 'off'),
}

#: Knobs whose best value we genuinely do NOT know, so a core sweep must measure them. Anything
#: absent is pinned via :data:`CODEGEN_PINNED` instead of being searched.
CORE_UNCERTAIN = ('implementation', 'const_scalar_abi')

#: Knobs with a nearly-always-right value, pinned in a core sweep and re-opened only by a broad one.
#: Each records WHY, so a surprising broad result traces back to a bad assumption rather than noise.
CODEGEN_PINNED: Dict[str, object] = {
    'scalar_emission_type': 'scalar',  # register, not a load/store through a length-1 array
    'explicit_copy': 'on',  # single-element copies become `=`, contiguous ones memcpy, not dace::CopyND
    'index_ctype': 'int64_t',  # narrowing is a niche win and an overflow risk on real shapes
    'loop_index_type': 'int64_t',  # 64-bit index arithmetic is native; per-loop `auto` buys nothing
    'index_fn_qualifier': 'inline_constexpr',  # C++20, so it folds at compile time; others only relax that
    'heap_ptr_restrict': 'restrict',  # never a pessimisation: DaCe's buffers genuinely do not alias
    'inline_full_array_nsdfg': True,  # removes a call boundary the optimizer cannot see past
    # Measured: 0.37x on 16 tiny nests, 1.14x on 6 heavy ones. Broad sweeps can still try it.
    'split_nsdfg_translation_units': False,
    # GPU analogue of the above (one .cu per top-level nest). CPU contract, and we offload loops, so
    # the shape it keys on should not arise. Named rather than omitted so the axis stays visible.
    'external_translation_units': False,
    # SOUNDNESS, not speed: `ne` is equivalent only when the stride divides the trip count exactly,
    # else the loop overshoots and runs away. Per-loop property, global knob -- only `lt` is safe.
    'loop_bound_cmp': 'lt',
}

#: The reduced codegen sweep: uncertain knobs searched, the rest pinned to a known-good value.
CORE_CODEGEN_AXES: Dict[str, Sequence] = {
    **{
        name: (value, )
        for name, value in CODEGEN_PINNED.items()
    },
    **{
        name: CODEGEN_AXES[name]
        for name in CORE_UNCERTAIN
    },
}

#: Ceiling on the variants ONE plan may enumerate; the arena builds each once per discovered
#: compiler, so with the usual two this is ~144 compilations. The full product of every knob is six
#: figures, which is not a sweep anyone runs.
VARIANT_BUDGET = 72

#: Order in which a broad sweep re-opens pinned knobs, most likely to matter first. It opens them
#: while the budget allows, so the bound holds by construction rather than by hoping.
BROAD_PRIORITY = (
    'scalar_emission_type',
    'explicit_copy',
    'inline_full_array_nsdfg',
    'split_nsdfg_translation_units',
    'index_ctype',
    'index_fn_qualifier',
    'loop_index_type',
    'heap_ptr_restrict',
    'loop_bound_cmp',
    'external_translation_units',
)


def broad_codegen_axes(budget: int = VARIANT_BUDGET) -> Dict[str, Sequence]:
    """Core axes widened by re-opening pinned knobs in :data:`BROAD_PRIORITY` order, within ``budget``."""
    axes = dict(CORE_CODEGEN_AXES)
    # Count the flag axes too: they multiply every codegen combination, so budgeting the codegen axes
    # alone overshoots by exactly that factor once the plan is assembled.
    total = 1
    for values in (*axes.values(), *FLAG_AXES.values()):
        total *= len(values)
    for name in BROAD_PRIORITY:
        widened = CODEGEN_AXES[name]
        if total * len(widened) > budget:
            continue
        axes[name] = widened
        total *= len(widened)
    return axes


class AgentMode(enum.Enum):
    """What an agent is asking us to do with its candidate."""
    # no __slots__: Enum
    #: The agent fixes part of the configuration; we sweep whatever it left open, around that point.
    SEARCH = 'search'
    #: The agent has already decided. Build exactly this, measure it, do not explore around it.
    EXACT = 'exact'


@dataclass(frozen=True, slots=True)
class AgentVariant:
    """A candidate an agent supplies directly, rather than one the sweep enumerates.

    An agent may hand over finished source, a flag set, or both. Either way the candidate gets the
    same oracle validation and the same timing as an enumerated variant, so it has to win on
    measurement rather than on being the agent's idea.

    :param label: name for the candidate in the report.
    :param mode: whether we explore around this point or build exactly it.
    :param source: replacement source to compile, or ``None`` to use the sweep's generated source.
    :param flags: exact compiler flags, or ``None`` to take them from ``axes``.
    :param axes: axis settings this candidate fixes; under ``SEARCH`` the rest stay open.
    """
    label: str
    mode: AgentMode = AgentMode.EXACT
    source: Optional[Union[str, Path]] = None
    flags: Optional[Sequence[str]] = None
    axes: Optional[Dict[str, object]] = None

    def supplies_code(self) -> bool:
        return self.source is not None

    def supplies_flags(self) -> bool:
        return self.flags is not None

    def span(self, sweep_axes: Dict[str, Sequence]) -> int:
        """Builds this candidate costs: ``EXACT`` one, ``SEARCH`` the product of the axes it left open."""
        if self.mode is AgentMode.EXACT:
            return 1
        fixed = set(self.axes or ())
        total = 1
        for name, values in sweep_axes.items():
            if name not in fixed:
                total *= len(values)
        return total


@dataclass(frozen=True, slots=True)
class SearchPlan:
    """The decision, before anything is compiled.

    :param kind: the disambiguated input kind.
    :param space: which axes may move.
    :param axes: axis name -> candidate values, the cartesian product the sweep will explore.
    :param needs_parse: whether a frontend must lower the input to an SDFG first.
    :param reason: why this space was chosen, carried into the report.
    """
    kind: InputKind
    space: SearchSpace
    axes: Dict[str, Sequence] = field(default_factory=dict)
    needs_parse: bool = False
    reason: str = ''
    agent_variants: Tuple[AgentVariant, ...] = ()

    def variant_count(self) -> int:
        """Variants the sweep enumerates, before any agent contribution."""
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    def total_count(self) -> int:
        """Everything that will be built: the enumerated sweep plus each agent candidate's span."""
        return self.variant_count() + sum(v.span(self.axes) for v in self.agent_variants)


def classify_input(source: Union[str, Path], kind: Optional[str] = None) -> InputKind:
    """Decide what ``source`` is.

    ``kind`` overrides the suffix and is the only way to force Fortran to compile-as-is: unforced, a
    ``.f90`` defaults to PARSING, whose space strictly contains the other (parsing still sweeps the
    flag axes; compiling as-is can never reach the codegen axes).

    :raises ValueError: if ``kind`` is unknown, or the suffix is unrecognised.
    """
    if kind is not None:
        try:
            return InputKind(kind)
        except ValueError:
            known = ', '.join(sorted(k.value for k in InputKind))
            raise ValueError(f'unknown input kind {kind!r}; expected one of: {known}') from None

    suffix = Path(source).suffix.lower()
    if suffix in PARSEABLE_SUFFIXES:
        return InputKind(PARSEABLE_SUFFIXES[suffix])
    if suffix in COMPILABLE_SUFFIXES:
        named = InputKind(COMPILABLE_SUFFIXES[suffix])
        return InputKind.FORTRAN_PARSE if named is InputKind.FORTRAN_SOURCE else named

    known = ', '.join(sorted(set(COMPILABLE_SUFFIXES) | set(PARSEABLE_SUFFIXES)))
    raise ValueError(f'cannot classify {source!r}: unrecognised suffix {suffix!r}. '
                     f'Known suffixes: {known}. Pass kind= to state it explicitly.')


def plan_search(source: Union[str, Path],
                kind: Optional[str] = None,
                agent_variants: Sequence[AgentVariant] = ()) -> SearchPlan:
    """Choose the search space for ``source``. Pure -- no toolchain, no filesystem beyond the suffix.

    :param agent_variants: candidates an agent supplies directly. ADDED to the sweep; they never
                           change which axes it moves, so steered and unsteered runs stay comparable.
    """
    resolved = classify_input(source, kind)
    extra = tuple(agent_variants)

    if resolved in PROVIDED_SOURCE_KINDS:
        return SearchPlan(kind=resolved,
                          space=SearchSpace.FLAGS,
                          axes=dict(FLAG_AXES),
                          needs_parse=False,
                          agent_variants=extra,
                          reason=f'{resolved.value} is finished source: the code is fixed, so only the '
                          'compiler invocation can vary')

    return SearchPlan(kind=resolved,
                      space=SearchSpace.CODEGEN,
                      axes={
                          **FLAG_AXES,
                          **broad_codegen_axes()
                      },
                      needs_parse=resolved is not InputKind.SDFG,
                      agent_variants=extra,
                      reason=f'{resolved.value} is lowered to an SDFG, so the generated code varies too; '
                      f'bounded at {VARIANT_BUDGET} variants per compiler')


def lower_to_sdfg(source: Union[str, Path], kind: InputKind) -> dace.SDFG:
    """Lower a parseable input to an SDFG.

    NumPy goes through the DaCe Python frontend, Fortran through ``dace_fortran``. Both imports are
    deferred so planning needs neither installed.

    :raises ImportError: if the frontend is missing.
    """
    path = Path(source)
    if kind is InputKind.SDFG:
        from dace.sdfg import SDFG
        return SDFG.from_file(str(path))

    if kind is InputKind.FORTRAN_PARSE:
        try:
            # RuntimeError as well as ImportError: dace_fortran.build detects an LLVM/Flang toolchain
            # AT IMPORT and raises RuntimeError('Cannot find LLVM...') when there is none, so catching
            # only ImportError lets a missing toolchain surface as a bare traceback from a dependency.
            from dace_fortran.build import make_builder
        except (ImportError, RuntimeError) as exc:
            raise ImportError('parsing Fortran needs the dace-fortran frontend AND an LLVM/Flang '
                              'toolchain it can find: install it editable (see requirements-dev.txt) '
                              f'after dace, and check the toolchain. Underlying cause: {exc}') from exc
        return make_builder(path.read_text(), name=path.stem).build()

    if kind is InputKind.NUMPY:
        raise NotImplementedError('numpy -> SDFG lowering is not wired yet; the plan still reports '
                                  'needs_parse so a caller sees the gap rather than a wrong answer')

    raise ValueError(f'{kind.value} is not a parseable kind')
