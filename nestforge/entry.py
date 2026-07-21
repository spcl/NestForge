# nest-forge entry contract: take whatever the user has, pick the search space, measure the variants.
"""One entry point over the 4-phase optimizer.

The premise is that language- and compiler-defined semantics are not to be trusted: whether a
vectorizer flag, an fp mode or a codegen path is faster is decided by compiling and timing it, never
by assuming. What we are ALLOWED to vary depends on what the caller gave us, so the input kind
selects the search space:

===============================  ===========================  ==================================
input                            what we can still change     search space
===============================  ===========================  ==================================
C / C++ / Fortran SOURCE         only how it is compiled      vectorizer x cost-model x fp x cc
NumPy / Fortran to be PARSED     the generated code too       codegen combos x flags x cc
anything, with no agent driving  everything                   every variant, exhaustively
===============================  ===========================  ==================================

Planning is pure: :func:`plan_search` inspects the input and returns a :class:`SearchPlan` without
touching a compiler, so the contract is testable on a machine with no toolchain. Execution lives in
:func:`optimize_program`, which hands the plan to the existing arena.
"""
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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
    C_SOURCE = 'c_source'
    CPP_SOURCE = 'cpp_source'
    FORTRAN_SOURCE = 'fortran_source'  # compile as-is
    FORTRAN_PARSE = 'fortran_parse'  # lower to an SDFG via dace-fortran
    NUMPY = 'numpy'
    SDFG = 'sdfg'


class SearchSpace(enum.Enum):
    """Which axes the sweep is allowed to move."""
    FLAGS = 'flags'  # case A -- the code is fixed, only the compiler invocation varies
    CODEGEN = 'codegen'  # case B -- we generate the C++, so codegen varies too
    ALL = 'all'  # case C -- no agent, so everything varies exhaustively


#: Kinds that arrive as finished source. These can never reach the codegen axes.
PROVIDED_SOURCE_KINDS = frozenset({InputKind.C_SOURCE, InputKind.CPP_SOURCE, InputKind.FORTRAN_SOURCE})
#: Kinds we lower to an SDFG ourselves.
PARSED_KINDS = frozenset({InputKind.NUMPY, InputKind.FORTRAN_PARSE, InputKind.SDFG})

#: Compiler-invocation axes. Values are resolved against the discovered toolchain at execution time;
#: naming them here keeps the contract inspectable without a compiler present.
FLAG_AXES: Dict[str, Sequence[str]] = {
    'vectorizer': ('none', 'auto', 'auto+width', 'slp'),
    'cost_model': ('default', 'unlimited', 'cheap'),
    'fp': ('strict', 'fast', 'associative'),
}

#: Every DaCe CPU codegen axis, with the values verified against dace `extended` config_schema.yml.
#: `implementation` is the old-vs-new switch the contract is built around; the rest are the readable
#: generator's knobs and are no-ops under `legacy`.
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
    'scalar_emission_type': ('scalar', 'keep'),
    'explicit_copy': ('on', 'off'),
}

#: Knobs whose best value we genuinely do NOT know, so a core sweep must measure them.
#: Anything absent from here is pinned to :data:`CODEGEN_PINNED` instead of being searched.
CORE_UNCERTAIN = ('implementation', 'const_scalar_abi', 'loop_bound_cmp')

#: Knobs with a value that is right nearly always, pinned in a core sweep and only opened up in an
#: exhaustive one. Each entry records WHY, so a surprising exhaustive result can be traced back to a
#: bad assumption here rather than looking like noise.
CODEGEN_PINNED: Dict[str, object] = {
    # A scalar beats a length-1 array essentially always: it lands in a register instead of forcing a
    # load/store through memory. This is also the repo's own standing rule for internal values.
    'scalar_emission_type': 'scalar',
    # Single-element copies collapse to `=` and contiguous ones to memcpy, instead of dace::CopyND.
    'explicit_copy': 'on',
    # 64-bit indices avoid overflow on real shapes; narrowing is a niche win and a correctness risk.
    'index_ctype': 'int64_t',
    # One index width everywhere on modern 64-bit hardware, rather than a per-loop `auto` decision:
    # 64-bit index arithmetic is native, so there is nothing to buy by varying it.
    'loop_index_type': 'int64_t',
    # We compile as C++20, so the index function is always constexpr -- it folds at compile time and
    # the other qualifiers only relax that. (`consteval` would be stronger still, forcing the fold,
    # but DaCe's schema offers no such value; inline_constexpr is the strongest available.)
    'index_fn_qualifier': 'inline_constexpr',
    # `restrict` on the heap pointers is never a pessimisation: it only ever tells the optimizer the
    # buffers do not alias, which is a fact of how DaCe allocates them.
    'heap_ptr_restrict': 'restrict',
    # Inlining a fully-passed array nested SDFG removes a call boundary the optimizer cannot see past.
    'inline_full_array_nsdfg': True,
    # Splitting translation units was MEASURED to lose on small nests (0.37x on 16 tiny nests) and win
    # only on heavy ones (1.14x on 6); off is the right default, exhaustive can still try it.
    'split_nsdfg_translation_units': False,
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


@dataclass(frozen=True)
class SearchPlan:
    """The decision, before anything is compiled.

    :param kind: the disambiguated input kind.
    :param space: which axes may move.
    :param axes: axis name -> candidate values, the cartesian product the sweep will explore.
    :param needs_parse: whether a frontend must lower the input to an SDFG first.
    :param reason: why this space was chosen, carried into the report so a run explains itself.
    """
    kind: InputKind
    space: SearchSpace
    axes: Dict[str, Sequence] = field(default_factory=dict)
    needs_parse: bool = False
    reason: str = ''

    def variant_count(self) -> int:
        """Size of the cartesian product -- the number of variants a full sweep will build."""
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total


def classify_input(source: Union[str, Path], kind: Optional[str] = None) -> InputKind:
    """Decide what ``source`` is.

    ``kind`` overrides the suffix and is the ONLY way to resolve the Fortran ambiguity in the
    direction of case A: a ``.f90`` can either be compiled as-is or lowered to an SDFG, and those
    select different search spaces. Unforced, Fortran defaults to PARSING, because that space
    strictly contains the other -- parsing still sweeps the flag axes, while compiling as-is can
    never reach the codegen axes.

    :raises ValueError: if ``kind`` is not a known kind, or the suffix is unrecognised.
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


def plan_search(source: Union[str, Path], kind: Optional[str] = None, agent: Optional[object] = None) -> SearchPlan:
    """Choose the search space for ``source``. Pure -- no toolchain, no filesystem beyond the suffix.

    :param agent: the driving optimizer, or ``None``. With no agent there is nothing to steer a
                  targeted search, so the contract falls back to sweeping every variant.
    """
    resolved = classify_input(source, kind)

    if resolved in PROVIDED_SOURCE_KINDS:
        return SearchPlan(kind=resolved,
                          space=SearchSpace.FLAGS,
                          axes=dict(FLAG_AXES),
                          needs_parse=False,
                          reason=f'{resolved.value} is finished source: the code is fixed, so only the '
                          'compiler invocation can vary')

    if agent is None:
        return SearchPlan(kind=resolved,
                          space=SearchSpace.ALL,
                          axes={
                              **FLAG_AXES,
                              **CODEGEN_AXES
                          },
                          needs_parse=resolved is not InputKind.SDFG,
                          reason='no agent to steer the search, so every variant is swept')

    return SearchPlan(kind=resolved,
                      space=SearchSpace.CODEGEN,
                      axes={
                          **FLAG_AXES,
                          **CORE_CODEGEN_AXES
                      },
                      needs_parse=resolved is not InputKind.SDFG,
                      reason=f'{resolved.value} is lowered to an SDFG, so the generated code varies too; '
                      'core knobs only -- knobs with a known-good value are pinned')


def lower_to_sdfg(source: Union[str, Path], kind: InputKind):
    """Lower a parseable input to an SDFG.

    NumPy goes through the DaCe Python frontend; Fortran through ``dace_fortran``. Both imports are
    deferred so that planning, and any caller that never parses, needs neither installed.

    :raises ImportError: with an actionable message if the frontend is missing.
    """
    path = Path(source)
    if kind is InputKind.SDFG:
        from dace.sdfg import SDFG
        return SDFG.from_file(str(path))

    if kind is InputKind.FORTRAN_PARSE:
        try:
            from dace_fortran.build import make_builder
        except ImportError as exc:
            raise ImportError('parsing Fortran needs the dace-fortran frontend: install it editable '
                              '(see requirements-dev.txt). Its pyproject pins dace @ FaCe, which is a '
                              'subset of extended, so an editable install resolves against the extended '
                              'checkout nest-forge already uses.') from exc
        return make_builder(path.read_text(), name=path.stem).build()

    if kind is InputKind.NUMPY:
        raise NotImplementedError('numpy -> SDFG lowering is not wired yet; the plan reports it as '
                                  'needs_parse so a caller can see the gap rather than get a wrong answer')

    raise ValueError(f'{kind.value} is not a parseable kind')
