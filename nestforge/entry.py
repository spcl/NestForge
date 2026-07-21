# nest-forge entry contract: take whatever the user has, pick the search space, measure the variants.
"""One entry point over the 4-phase optimizer.

The premise is that language- and compiler-defined semantics are not to be trusted: whether a
vectorizer flag, an fp mode or a codegen path is faster is decided by compiling and timing it, never
by assuming. What we are ALLOWED to vary depends only on what the caller gave us:

=============================  =========================  ====================================
input                          what we can still change   search space
=============================  =========================  ====================================
C / C++ / Fortran SOURCE       only how it is compiled    vectorize x fp                    (9)
NumPy / Fortran / SDFG         the generated code too     + codegen knobs, budgeted        (72)
=============================  =========================  ====================================

per compiler; the arena builds each variant once per discovered toolchain.

An agent does NOT change the space. It CONTRIBUTES: an :class:`AgentVariant` may carry finished
source, an exact flag set, or both, and is measured on the same footing as every enumerated variant.
Keeping the space fixed is deliberate -- a steered run and an unsteered run then cover identical
ground, so the agent's contribution is what the difference in outcome actually measures.

Planning is pure: :func:`plan_search` inspects the input and returns a :class:`SearchPlan` without
touching a compiler, so the contract is testable on a machine with no toolchain. Execution lives in
:func:`optimize_program`, which hands the plan to the existing arena.
"""
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    """Which axes the sweep is allowed to move.

    There are only two, and an agent does not change which one applies: the input decides what we may
    vary, and an agent CONTRIBUTES candidates (its own code, its own flags) on top of that. A steered
    run and an unsteered run therefore sweep exactly the same space, which is what makes their
    results comparable.
    """
    FLAGS = 'flags'  # the code is fixed, only the compiler invocation varies
    CODEGEN = 'codegen'  # we generate the C++, so codegen varies too


#: Kinds that arrive as finished source. These can never reach the codegen axes.
PROVIDED_SOURCE_KINDS = frozenset({InputKind.C_SOURCE, InputKind.CPP_SOURCE, InputKind.FORTRAN_SOURCE})
#: Kinds we lower to an SDFG ourselves.
PARSED_KINDS = frozenset({InputKind.NUMPY, InputKind.FORTRAN_PARSE, InputKind.SDFG})

#: Compiler-invocation axes. Values are resolved to per-compiler flags at execution time; naming them
#: here keeps the contract inspectable without a compiler present.
#:
#: `vectorize` is ONE axis, not a vectorizer switch crossed with a cost model. Those two are not
#: independent: with vectorization off the cost model has nothing to decide, so the crossed form
#: emitted the same variant three times over. The three values are the distinct outcomes -- do not
#: vectorize, vectorize only where it is cheaply profitable, vectorize wherever the compiler will.
#:
#: `fp` mirrors :data:`nestforge.arena.FP_MODES`, which owns the actual flag lists and the matching
#: comparison tolerance per mode (bit-exact for ieee-strict, looser as the mode relaxes). Restated
#: here so planning stays free of heavy imports; a test asserts the two never drift apart.
FLAG_AXES: Dict[str, Sequence[str]] = {
    'vectorize': ('none', 'cheap', 'auto'),
    'fp': ('ieee-strict', 'fast-but-ieee', 'fast-math'),
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
    'external_translation_units': (False, True),
    'scalar_emission_type': ('scalar', 'keep'),
    'explicit_copy': ('on', 'off'),
}

#: Knobs whose best value we genuinely do NOT know, so a core sweep must measure them.
#: Anything absent from here is pinned to :data:`CODEGEN_PINNED` instead of being searched.
CORE_UNCERTAIN = ('implementation', 'const_scalar_abi')

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
    # The GPU analogue of the above: lifts each top-level GPU nest into a standalone SDFG with its own
    # .cu. Off -- this contract targets CPU, and because we OFFLOAD loops the shape it keys on should
    # not arise here at all. Named rather than omitted so a caller can see the axis exists.
    'external_translation_units': False,
    # `i < N` rather than `i != N`. This is a SOUNDNESS pin, not a speed one: `ne` is equivalent only
    # when the stride divides the trip count exactly, and otherwise the loop overshoots its bound and
    # runs away. That is a per-loop property and this knob is global, so the safe form is the only one
    # that can be set globally.
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

#: Ceiling on the variants ONE plan may enumerate. The arena then builds each variant once per
#: discovered compiler, so with the usual two this is ~144 compilations -- a sweep that finishes.
#: The full cartesian product of every knob is six figures, which is not a sweep anyone runs.
VARIANT_BUDGET = 72

#: Which pinned knobs a broad sweep re-opens first, most likely to matter first. A broad sweep walks
#: this list and opens knobs while the budget allows, so the bound is respected by construction
#: rather than by hoping the product stays small.
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
    """Core axes widened by re-opening pinned knobs in :data:`BROAD_PRIORITY` order, within ``budget``.

    Used when no agent is steering: with nothing to direct a targeted search the sweep should cover
    more ground than core, but "everything" is six figures of builds, so it is bounded here instead.
    """
    axes = dict(CORE_CODEGEN_AXES)
    # Count the flag axes too: they multiply every codegen combination, so a budget applied to the
    # codegen axes alone would be overshot by exactly that factor once the plan is assembled.
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
    #: The agent fixes part of the configuration; we sweep whatever it left open, around that point.
    SEARCH = 'search'
    #: The agent has already decided. Build exactly this, measure it, do not explore around it.
    EXACT = 'exact'


@dataclass(frozen=True)
class AgentVariant:
    """A candidate an agent supplies directly, rather than one the sweep enumerates.

    An agent is not restricted to steering knobs: it may hand over finished source, a flag set, or
    both. Either way the candidate is measured on the same footing as every enumerated variant --
    same validation against the oracle, same timing -- so an agent's idea has to win on measurement
    rather than on being the agent's idea.

    The two modes answer different questions. ``SEARCH`` says "start here and explore what I left
    unspecified", which is how an agent narrows a space it does not fully understand. ``EXACT`` says
    "I have decided", which is how an agent that has already done the reasoning avoids paying for a
    sweep it does not need.

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
        """How many builds this candidate costs.

        ``EXACT`` is one. ``SEARCH`` is the product of every axis the agent left open -- it pins what
        it named and we explore the remainder.
        """
        if self.mode is AgentMode.EXACT:
            return 1
        fixed = set(self.axes or ())
        total = 1
        for name, values in sweep_axes.items():
            if name not in fixed:
                total *= len(values)
        return total


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
    agent_variants: Tuple[AgentVariant, ...] = ()

    def variant_count(self) -> int:
        """Variants the sweep enumerates, before any agent contribution."""
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    def total_count(self) -> int:
        """Everything that will be built: the enumerated sweep plus what each agent candidate costs.

        An EXACT candidate costs one build; a SEARCH candidate costs the product of the axes it left
        open, since we explore around the point it fixed.
        """
        return self.variant_count() + sum(v.span(self.axes) for v in self.agent_variants)


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


def plan_search(source: Union[str, Path],
                kind: Optional[str] = None,
                agent_variants: Sequence[AgentVariant] = ()) -> SearchPlan:
    """Choose the search space for ``source``. Pure -- no toolchain, no filesystem beyond the suffix.

    :param agent_variants: candidates an agent supplies directly (finished code, exact flags, or
                           both). They are ADDED to the sweep; they never change which axes it moves,
                           so a steered and an unsteered run cover the same space and stay comparable.
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
