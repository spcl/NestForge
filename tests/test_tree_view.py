# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ASCII tree the agent reads (:func:`nestforge.ir.introspect.describe_graph`).

The tree is the agent's whole view of the program, so its format is an interface: a change to it
changes what every agent prompt sees. The golden test below pins it in full, so a format change has
to be made deliberately rather than drifting out of an unrelated edit.
"""

import re

import numpy as np

import pytest

import dace as dc

from nestforge.ir import introspect
from nestforge.ir.introspect import describe_graph, interstate_definitions, kernel_body, resolve_scalars
from nestforge.ir.names import normalize_for_tree
from nestforge.session import Session


@dc.program
def shaped(A: dc.float64[20], B: dc.float64[20], out: dc.float64[20]):
    """One map nest, a loop around a conditional, and a scalar statement -- one of each thing the tree
    has a line shape for."""
    for i in dc.map[0:20]:
        B[i] = A[i] + 1.0
    s = B[0] * 2.0
    for i in range(1, 20):
        if A[i] > 0.0:
            out[i] = B[i] * s
        else:
            out[i] = -B[i]


GOLDEN = """\
SDFG 'shaped'
|- state0_0
|  |- kernel1_0  [i0=0:20]  reads=['A'] writes=['B']
|  `- kernel1_1  [__nf_wrap=0:1]  reads=['s0'] writes=['s1']
`- for0_0  i=0:19
   |- state1_0
   `- if1_0
      |- block2_0  when A[i + 1] > 0.0
      |  `- state3_0
      |     |- kernel4_0  [__nf_wrap=0:1]  reads=['s1', 's2'] writes=['s3']
      |     `- kernel4_1  [__nf_wrap=0:1]  reads=['s3'] writes=['out']
      `- block2_1  else
         `- state3_1
            |- kernel4_2  [__nf_wrap=0:1]  reads=['s4'] writes=['s5']
            `- kernel4_3  [__nf_wrap=0:1]  reads=['s5'] writes=['out']
"""


def tree_of(program) -> str:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = "shaped"
    normalize_for_tree(sdfg)
    return describe_graph(sdfg)


def test_the_tree_format_is_pinned():
    """If this fails, the format changed. Update GOLDEN only when that change is the intent -- every
    agent prompt reads this shape."""
    assert tree_of(shaped) + "\n" == GOLDEN


def test_every_line_is_a_guide_then_one_labelled_thing():
    for line in tree_of(shaped).splitlines()[1:]:
        assert re.match(
            r"^(\|  |   )*(\|- |`- )"
            r"(state|for|while|if|block|continue|break|return|kernel)\d+_\d+\b",
            line,
        ), line


def test_indentation_tracks_the_level_in_the_label():
    """A line's depth in the tree and the level in its canonical name are the same number, so the
    agent can read either one."""
    for line in tree_of(shaped).splitlines()[1:]:
        guide, body = re.match(r"^((?:\|  |   )*(?:\|- |`- ))(.*)$", line).groups()
        level = int(re.match(r"^[a-z]+(\d+)_", body).group(1))
        assert len(guide) == 3 * (level + 1), f"level {level} at guide width {len(guide)}: {line}"


def test_every_row_the_session_prints_is_a_label_its_labeled_calls_take():
    """Reading the tree and acting on it use one vocabulary: each row's first word is a key of the row index."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    labels = [re.sub(r"^[|` -]*", "", line).split()[0] for line in session.describe().splitlines()[1:]]
    assert labels and all(label in session.row_index() for label in labels), labels


# conditions read as the arrays they test


def test_a_condition_names_the_array_it_really_reads():
    """The frontend hoists the scalar read to an interstate assignment (`A_index = A[1 + i]`) and the
    branch then tests a name that means nothing on its own."""
    tree = tree_of(shaped)
    assert "when A[i + 1] > 0.0" in tree, tree
    assert "A_index" not in tree


def test_a_name_with_two_definitions_is_left_alone():
    """Which definition reaches a block depends on the path taken, so folding either one in would show
    a condition the program does not always evaluate."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    assert "A_index" in interstate_definitions(sdfg), "fixture no longer hoists the scalar read"
    # Give the same name a second, different definition on another edge -- as two branches assigning
    # one variable do.
    edge = next(
        e
        for cfg in sdfg.all_control_flow_regions(recursive=True)
        for e in cfg.edges()
        if "A_index" in e.data.assignments
    )
    other = next(e for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges() if e is not edge)
    other.data.assignments["A_index"] = "A[0]"
    assert "A_index" not in interstate_definitions(sdfg)
    assert "A_index" in describe_graph(sdfg), "an ambiguous name must stay unresolved, not vanish"


def test_resolution_terminates_on_a_self_referential_definition():
    """`i = i + 1` on a back edge is ordinary; substituting it forever is not."""
    assert resolve_scalars("i < N", {"i": "i + 1"}) == "i + 1 < N"


def test_resolution_follows_a_chain_to_the_array():
    assert resolve_scalars("c > 0", {"c": "b * 2", "b": "A[k]"}) == "A[k] * 2 > 0"


def test_an_unparsable_condition_is_passed_through():
    assert resolve_scalars("this is not python", {}) == "this is not python"


# bodies: what each kernel computes


@dc.program
def nested_maps(A: dc.float64[8, 4], B: dc.float64[8, 4]):
    """An outer kernel whose body is another kernel."""
    for i in dc.map[0:8]:
        for j in dc.map[0:4]:
            B[i, j] = A[i, j] * 2.0


def with_bodies(program) -> str:
    sdfg = program.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    return describe_graph(sdfg, bodies=True)


def test_an_index_is_simplified_even_when_nothing_is_defined():
    """Whether a program assigns anything elsewhere must not change how one condition prints."""
    assert resolve_scalars("A[(1 + (1 * i))] > 0.0", {}) == "A[i + 1] > 0.0"


def test_a_nested_sdfg_assignment_does_not_define_an_outer_name():
    """A nested SDFG has its own symbol namespace, so its ``k = 7`` says nothing about the outer ``k``."""
    inner = dc.SDFG("inner")
    first = inner.add_state("first", is_start_block=True)
    inner.add_state_after(first, "second", assignments={"k": "7"})
    outer = dc.SDFG("outer")
    outer.add_symbol("k", dc.int64)
    outer.add_state("call", is_start_block=True).add_nested_sdfg(inner, {}, {})

    assert "k" not in interstate_definitions(outer)


@pytest.mark.parametrize(
    ("init", "condition", "update", "want"),
    [
        ("i = 0", "i < 10", "i = i + 1", "i=0:10"),
        ("i = 9", "i >= 0", "i = i - 1", "i=9:-1:-1"),
    ],
    ids=["ascending", "descending"],
)
def test_a_loop_range_ends_one_past_its_last_value_in_its_direction(init, condition, update, want):
    """A descending loop printed with ``end + 1`` reads as stopping two values early."""
    from dace.sdfg.state import LoopRegion

    loop = LoopRegion("loop", condition, "i", init, update)

    assert introspect.loop_domain(loop, {}) == want


def test_bodies_are_off_by_default():
    """An emit per kernel is not free, and the structure alone is what a fusion decision needs."""
    assert not [line for line in tree_of(shaped).splitlines() if introspect.BODY in line]
    assert [line for line in with_bodies(shaped).splitlines() if introspect.BODY in line], "nothing to be off"


def test_a_leaf_kernel_prints_what_it_computes():
    tree = with_bodies(shaped)
    assert f"{introspect.BODY}B[i0] = " in tree, tree
    for line in tree.splitlines():
        if introspect.BODY in line:
            assert line.split(introspect.BODY, 1)[0].strip("| `") == "", "a body line must sit under its kernel"


def test_a_body_does_not_repeat_its_headers():
    """The kernel line already shows the domain those `for` headers iterate."""
    for line in with_bodies(shaped).splitlines():
        if introspect.BODY in line:
            assert not line.split(introspect.BODY, 1)[1].startswith("for "), line


def test_a_kernel_containing_a_kernel_has_no_body_of_its_own():
    """`map_body_lines` recurses into a nested map, and the tree already gives that map its own row,
    so emitting at both levels would print the inner kernel twice."""
    sdfg = nested_maps.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    nested = [
        (st, n)
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry)
        and any(isinstance(c, dc.sdfg.nodes.MapEntry) for c in st.scope_children()[n])
    ]
    assert nested, "fixture no longer nests one kernel inside another"
    for state, entry in nested:
        assert kernel_body(state, sdfg, entry, state.scope_children()) == []
    # and the inner one, which is a leaf, does carry the statement
    inner = [
        (st, n)
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry) and st.entry_node(n) is not None
    ]
    assert any(kernel_body(st, sdfg, n, st.scope_children()) for st, n in inner), "inner printed nothing"


def test_an_emitter_refusal_is_reported_on_the_line_not_raised(monkeypatch):
    """The tree is a read-only view. A nest the numpy projection cannot express is exactly what the
    agent needs to be told about, so it must not take the whole tree down."""

    def refuse(state, sdfg, entry):
        raise introspect.UnsupportedNest("no emitter for this")

    monkeypatch.setattr(introspect, "map_body_lines", refuse)
    tree = with_bodies(shaped)
    assert "<not emitted: no emitter for this>" in tree


# reductions on the kernel line


@dc.program
def matvec(A: dc.float64[8, 4], B: dc.float64[4], C: dc.float64[8]):
    """A WCR over one of two map axes -- the classic tree reduction."""
    for i, j in dc.map[0:8, 0:4]:
        C[i] += A[i, j] * B[j]


def test_a_reduction_is_named_on_the_kernel_line():
    """A WCR on a map IS a tree reduction: the map declares its iterations independent, so the fold
    order is unspecified and a backend may use a register accumulator or an OpenMP clause. That is
    structural, and the agent should not have to read the body to find it."""
    tree = tree_of(matvec)
    assert "reduce=(+ over i1 -> C)" in tree, tree


def test_only_the_collapsed_axis_is_reported():
    """The reduced axes are the map parameters the output subset does not mention -- a map over
    (i0, i1) writing C[i0] has collapsed i1 and only i1."""
    tree = tree_of(matvec)
    assert "over i1 ->" in tree and "over i0" not in tree, tree


def test_a_kernel_without_a_reduction_says_nothing():
    assert "reduce=" not in tree_of(shaped)


def test_the_reduction_op_is_read_off_the_wcr():
    sdfg = matvec.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    state, entry = next(
        (st, n) for st in sdfg.all_states() for n in st.nodes() if isinstance(n, dc.sdfg.nodes.MapEntry)
    )
    assert introspect.kernel_reductions(state, entry) == ["+ over i1 -> C"]
    # A different op reads as itself, not as "+".
    exit_node = state.exit_node(entry)
    edge = next(e for e in state.in_edges(exit_node) if e.data.wcr is not None)
    edge.data.wcr = "lambda x, y: max(x, y)"
    assert introspect.kernel_reductions(state, entry) == ["max over i1 -> C"]


def test_a_body_is_not_recovered_by_slicing_the_emitted_block():
    """The body comes from `map_body_lines`, not from dropping len(params) lines off `map_lines`
    and dedenting by 4 * len(params). That arithmetic held only while every header was exactly one
    line and every body line carried the full indent."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    state, entry = next(
        (st, n)
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry) and st.entry_node(n) is None
    )
    body = kernel_body(state, sdfg, entry, state.scope_children())
    assert body, "the fixture kernel emits nothing"
    for line in body:
        assert not line.startswith(" "), f"a body line arrived still indented: {line!r}"
        assert not line.startswith("for "), f"a header leaked into the body: {line!r}"
    # and it agrees with the full block the emitter produces for the same kernel
    full = introspect.map_body_lines(state, sdfg, entry)
    assert body == full


# one kernel's body


def first_nest(program):
    sdfg = program.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    state = next(st for st in sdfg.all_states() if any(isinstance(n, dc.nodes.MapEntry) for n in st.nodes()))
    entry = next(n for n in state.scope_children()[None] if isinstance(n, dc.nodes.MapEntry))
    return sdfg, state, entry


def test_a_kernel_body_is_the_statements_without_their_headers():
    sdfg, state, entry = first_nest(shaped)
    body = kernel_body(state, sdfg, entry, state.scope_children())
    assert body and all(isinstance(line, str) for line in body)
    assert not any(line.startswith("for ") for line in body), "headers are on the kernel line already"


def test_a_reduction_body_is_folded():
    """An explicit accumulate is the only point rendering of a reduction; `np.sum` is a whole-array
    spelling that belongs to the slice form."""
    sdfg, state, entry = first_nest(matvec)
    body = "\n".join(kernel_body(state, sdfg, entry, state.scope_children()))
    assert "C[i0] = C[i0] +" in body, body
    assert "np.sum" not in body


# a kernel's representation: pure, runnable numpy


def source_of_first_kernel(program):
    sdfg, state, entry = first_nest(program)
    return sdfg, state, introspect.kernel_source(state, sdfg, entry)


def test_a_kernel_source_is_a_whole_module_not_a_fragment():
    """`kernel_body` is the excerpt the tree prints; its statements reference loop variables that only
    exist inside their headers. The representation has to be something an agent can run."""
    _, _, source = source_of_first_kernel(shaped)
    assert source.startswith("import numpy as np")
    assert "\ndef kernel" in source
    compile(source, "<kernel>", "exec")  # syntactically a module, not a snippet


def test_a_kernel_source_runs_with_nothing_injected():
    """No EMITTED_BUILTINS, no `np` handed in -- pure numpy or it does not count."""
    _, _, source = source_of_first_kernel(matvec)
    namespace = {}
    exec(source, namespace)  # a bare dict: only what the source itself defines
    assert "int_floor" in namespace and "np" in namespace


def test_a_kernel_source_computes_what_the_sdfg_computes():
    """The point of it being runnable: emit, execute, compare. A representation nothing can execute
    cannot be shown to be correct."""
    sdfg, _, source = source_of_first_kernel(matvec)
    namespace = {}
    exec(source, namespace)
    kernel = next(v for k, v in namespace.items() if k.startswith("kernel") and callable(v))

    A = np.linspace(0.5, 4.0, 32).reshape(8, 4).copy()
    B = np.linspace(1.0, 2.0, 4).copy()
    from_source = np.zeros(8)
    kernel(A, B, from_source)

    from_sdfg = np.zeros(8)
    # NumPy never contracts a*b+c into an FMA; without this the SDFG's result depends on the host CPU.
    strict_args = dc.Config.get("compiler", "cpu", "args") + " -ffp-contract=off"
    with dc.config.set_temporary("compiler", "cpu", "args", value=strict_args):
        sdfg(A=A.copy(), B=B.copy(), C=from_sdfg)
    assert np.array_equal(from_source, from_sdfg), (from_source, from_sdfg)


N, M = dc.symbol("N", dtype=dc.int64), dc.symbol("M", dtype=dc.int64)


@dc.program
def last_column(A: dc.float64[N, M], B: dc.float64[N]):
    for i in dc.map[0:N]:
        B[i] = A[i, M - 1]


def test_a_kernel_source_takes_a_symbol_only_its_body_reads():
    """``M`` bounds no map, yet the body indexes with it, so the signature must take it."""
    _, _, source = source_of_first_kernel(last_column)
    namespace = {}
    exec(source, namespace)
    kernel = next(v for k, v in namespace.items() if k.startswith("kernel") and callable(v))
    A, B = np.arange(12, dtype=np.float64).reshape(3, 4).copy(), np.zeros(3)

    kernel(A=A, B=B, M=4, N=3)

    np.testing.assert_array_equal(B, A[:, 3])
