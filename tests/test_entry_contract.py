# Contract tests for nestforge.entry: input kind -> search space. No compiler required.
import itertools

import pytest

from nestforge.entry import (CODEGEN_AXES, CODEGEN_PINNED, COMPILABLE_SUFFIXES, CORE_CODEGEN_AXES, CORE_UNCERTAIN,
                             FLAG_AXES, PARSEABLE_SUFFIXES, PARSED_KINDS, PROVIDED_SOURCE_KINDS, InputKind, SearchSpace,
                             classify_input, lower_to_sdfg, plan_search)


class Agent:
    """Stand-in for a driving optimizer -- plan_search only checks presence, never calls it."""
    name = 'stub'


# ---------------------------------------------------------------- classification


@pytest.mark.parametrize('suffix,expected', [
    ('.c', InputKind.C_SOURCE),
    ('.cc', InputKind.CPP_SOURCE),
    ('.cpp', InputKind.CPP_SOURCE),
    ('.cxx', InputKind.CPP_SOURCE),
    ('.py', InputKind.NUMPY),
    ('.sdfg', InputKind.SDFG),
    ('.sdfgz', InputKind.SDFG),
])
def test_suffix_classifies(suffix, expected):
    assert classify_input(f'kernel{suffix}') is expected


@pytest.mark.parametrize('suffix', ['.f', '.f90', '.f03', '.f08'])
def test_fortran_defaults_to_parsing(suffix):
    """Unforced Fortran parses: that space strictly contains compiling it as-is."""
    assert classify_input(f'kernel{suffix}') is InputKind.FORTRAN_PARSE


@pytest.mark.parametrize('suffix', ['.f', '.f90'])
def test_fortran_can_be_forced_to_source(suffix):
    assert classify_input(f'kernel{suffix}', kind='fortran_source') is InputKind.FORTRAN_SOURCE


def test_suffix_is_case_insensitive():
    assert classify_input('KERNEL.F90') is InputKind.FORTRAN_PARSE
    assert classify_input('KERNEL.CPP') is InputKind.CPP_SOURCE


@pytest.mark.parametrize('kind', [k.value for k in InputKind])
def test_explicit_kind_overrides_suffix(kind):
    """A .c path forced to any kind yields that kind -- the override is absolute."""
    assert classify_input('kernel.c', kind=kind) is InputKind(kind)


def test_unknown_suffix_is_rejected():
    with pytest.raises(ValueError, match='unrecognised suffix'):
        classify_input('kernel.rs')


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match='unknown input kind'):
        classify_input('kernel.c', kind='pascal')


def test_error_message_lists_known_suffixes():
    with pytest.raises(ValueError) as exc:
        classify_input('kernel.rs')
    for suffix in ('.c', '.cpp', '.f90', '.py'):
        assert suffix in str(exc.value)


# ---------------------------------------------------------------- the dispatch matrix


@pytest.mark.parametrize('kind', sorted(k.value for k in PROVIDED_SOURCE_KINDS))
@pytest.mark.parametrize('agent', [None, Agent()])
def test_provided_source_is_flags_only_regardless_of_agent(kind, agent):
    """Case A: the code is fixed, so no agent can unlock the codegen axes."""
    plan = plan_search(f'kernel.src', kind=kind, agent=agent)
    assert plan.space is SearchSpace.FLAGS
    assert not plan.needs_parse
    assert set(plan.axes) == set(FLAG_AXES)
    assert not set(plan.axes) & set(CODEGEN_AXES)


@pytest.mark.parametrize('kind', sorted(k.value for k in PARSED_KINDS))
def test_parsed_input_with_agent_searches_codegen(kind):
    """Case B: we generate the C++, so old-vs-new codegen is in play."""
    plan = plan_search('kernel.src', kind=kind, agent=Agent())
    assert plan.space is SearchSpace.CODEGEN
    assert 'implementation' in plan.axes
    assert set(plan.axes) >= set(FLAG_AXES)


@pytest.mark.parametrize('kind', sorted(k.value for k in PARSED_KINDS))
def test_parsed_input_without_agent_sweeps_everything(kind):
    """Case C: nothing to steer the search, so sweep all variants."""
    plan = plan_search('kernel.src', kind=kind, agent=None)
    assert plan.space is SearchSpace.ALL
    assert set(plan.axes) == set(FLAG_AXES) | set(CODEGEN_AXES)


def test_every_kind_is_dispatched():
    """No InputKind may fall through the contract unhandled."""
    for kind, agent in itertools.product(InputKind, [None, Agent()]):
        plan = plan_search('kernel.src', kind=kind.value, agent=agent)
        assert plan.space in SearchSpace
        assert plan.reason


def test_partition_is_total_and_disjoint():
    assert PROVIDED_SOURCE_KINDS | PARSED_KINDS == set(InputKind)
    assert not PROVIDED_SOURCE_KINDS & PARSED_KINDS


# ---------------------------------------------------------------- parsing flag


@pytest.mark.parametrize('kind,expected', [
    (InputKind.NUMPY, True),
    (InputKind.FORTRAN_PARSE, True),
    (InputKind.SDFG, False),
    (InputKind.C_SOURCE, False),
])
def test_needs_parse(kind, expected):
    assert plan_search('kernel.src', kind=kind.value, agent=Agent()).needs_parse is expected


# ---------------------------------------------------------------- the axes themselves


def test_flag_axes_cover_what_the_contract_names():
    """The contract names vectorizer, cost-model and fp flags explicitly."""
    assert {'vectorizer', 'cost_model', 'fp'} <= set(FLAG_AXES)
    for values in FLAG_AXES.values():
        assert len(values) >= 2, 'an axis with one value is not a search axis'


def test_codegen_axes_include_the_old_new_switch():
    """Old codegen vs new codegen is the axis the contract is built around."""
    assert 'legacy' in CODEGEN_AXES['implementation']
    assert 'experimental_readable' in CODEGEN_AXES['implementation']


def test_codegen_axis_values_are_distinct():
    for name, values in CODEGEN_AXES.items():
        assert len(set(values)) == len(values), f'{name} repeats a value'


def test_variant_count_is_the_cartesian_product():
    plan = plan_search('kernel.py', agent=Agent())
    expected = 1
    for values in plan.axes.values():
        expected *= len(values)
    assert plan.variant_count() == expected
    assert plan.variant_count() > 1


def test_all_space_is_at_least_as_large_as_codegen():
    """Case C must never search less than case B."""
    everything = plan_search('kernel.py', agent=None)
    directed = plan_search('kernel.py', agent=Agent())
    assert everything.variant_count() >= directed.variant_count()


def test_flags_space_is_smaller_than_parsed_space():
    """Handing us finished source genuinely costs search space -- that is the point of the contract."""
    provided = plan_search('kernel.c')
    parsed = plan_search('kernel.py')
    assert provided.variant_count() < parsed.variant_count()


def test_plan_is_immutable():
    plan = plan_search('kernel.c')
    with pytest.raises(Exception):
        plan.space = SearchSpace.ALL


def test_axes_are_copied_not_aliased():
    """Mutating a plan's axes must not corrupt the module-level axis table."""
    plan = plan_search('kernel.c')
    plan.axes['vectorizer'] = ('bogus', )
    assert FLAG_AXES['vectorizer'] != ('bogus', )


def test_suffix_tables_do_not_overlap_except_fortran():
    overlap = set(COMPILABLE_SUFFIXES) & set(PARSEABLE_SUFFIXES)
    assert not overlap, f'a suffix in both tables is ambiguous: {overlap}'


# ---------------------------------------------------------------- lowering

# ---------------------------------------------------------------- core vs exhaustive knobs


def test_core_and_pinned_partition_every_codegen_knob():
    """No knob may be silently absent from both the searched and the pinned set."""
    assert set(CORE_UNCERTAIN) | set(CODEGEN_PINNED) == set(CODEGEN_AXES)
    assert not set(CORE_UNCERTAIN) & set(CODEGEN_PINNED)


def test_pinned_values_are_legal_values_of_their_axis():
    for name, value in CODEGEN_PINNED.items():
        assert value in CODEGEN_AXES[name], f'{name} pinned to {value!r}, not one of {CODEGEN_AXES[name]}'


def test_core_pins_the_knobs_we_are_confident_about():
    core = plan_search('kernel.py', agent=Agent())
    for name in CODEGEN_PINNED:
        assert len(core.axes[name]) == 1, f'{name} should be pinned in a core sweep'


def test_core_searches_the_knobs_we_are_unsure_about():
    core = plan_search('kernel.py', agent=Agent())
    for name in CORE_UNCERTAIN:
        assert len(core.axes[name]) > 1, f'{name} is uncertain, a core sweep must measure it'


def test_scalar_beats_len1_array_so_it_is_pinned():
    """The worked example: we are ~certain, so do not spend variants on it in a core sweep."""
    assert CODEGEN_PINNED['scalar_emission_type'] == 'scalar'


def test_const_scalar_abi_is_searched_because_we_do_not_know():
    """The other worked example: by_ref vs by_value is genuinely open, so core must measure it."""
    assert 'const_scalar_abi' in CORE_UNCERTAIN
    assert set(CODEGEN_AXES['const_scalar_abi']) == {'by_ref', 'by_value'}


def test_old_vs_new_codegen_is_always_searched():
    assert 'implementation' in CORE_UNCERTAIN


def test_restrict_is_always_emitted():
    """Emitting restrict is never a pessimisation, so it is a pin and never a search axis."""
    assert CODEGEN_PINNED['heap_ptr_restrict'] == 'restrict'
    assert 'heap_ptr_restrict' not in CORE_UNCERTAIN


def test_index_function_is_always_constexpr():
    """We compile as C++20, so the index function always folds at compile time."""
    assert CODEGEN_PINNED['index_fn_qualifier'] == 'inline_constexpr'


def test_one_index_width_everywhere():
    """64-bit index arithmetic is native on modern hardware; nothing to buy by varying it."""
    assert CODEGEN_PINNED['loop_index_type'] == 'int64_t'
    assert CODEGEN_PINNED['index_ctype'] == 'int64_t'


def test_core_is_dramatically_smaller_than_exhaustive():
    core = plan_search('kernel.py', agent=Agent())
    everything = plan_search('kernel.py', agent=None)
    assert core.variant_count() < everything.variant_count()
    assert everything.variant_count() / core.variant_count() > 10


def test_exhaustive_opens_every_pinned_knob():
    everything = plan_search('kernel.py', agent=None)
    for name in CODEGEN_PINNED:
        assert len(everything.axes[name]) == len(CODEGEN_AXES[name])


def test_every_axis_value_set_is_non_empty():
    for table in (FLAG_AXES, CODEGEN_AXES, CORE_CODEGEN_AXES):
        for name, values in table.items():
            assert len(values) >= 1, f'{name} has no values'


# ---------------------------------------------------------------- lowering


def test_lower_rejects_a_non_parseable_kind():
    with pytest.raises(ValueError, match='not a parseable kind'):
        lower_to_sdfg('kernel.c', InputKind.C_SOURCE)


def test_numpy_lowering_reports_the_gap_loudly():
    """Better an explicit NotImplementedError than a silently wrong answer."""
    with pytest.raises(NotImplementedError):
        lower_to_sdfg('kernel.py', InputKind.NUMPY)
