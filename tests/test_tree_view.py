# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ASCII tree the agent reads (:func:`nestforge.ir.introspect.describe_graph`).

The tree is the agent's whole view of the program, so its format is an interface: a change to it
changes what every agent prompt sees. The golden test below pins it in full, so a format change has
to be made deliberately rather than drifting out of an unrelated edit.
"""

import re
from pathlib import Path
from typing import Any, Never

import dace as dc
import numpy as np
import pytest
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.state import LoopRegion, SDFGState
from helpers import kernel_body

from nestforge.ir import introspect
from nestforge.ir.emit_python import UnsupportedNest
from nestforge.ir.introspect import describe_graph, interstate_definitions, resolve_scalars
from nestforge.ir.names import normalize_labels, top_level_nsdfgs
from nestforge.session import Session


@dc.program
def shaped(A: dc.float64[20], B: dc.float64[20], out: dc.float64[20]) -> None:
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
|- state0  reads=['A', 'B'] writes=['B', 's']  work=21 depth=2
|  `- doall0  [i=0:20]  reads=['A'] writes=['B']  work=20 depth=1
`- for0  i=1:20  num_states=3  work=19 depth=19
   |- state1
   `- if0
      |- block0  when A[i] > 0.0
      |  `- state2  reads=['B', 's'] writes=['out']  work=1 depth=1
      `- block1  else
         `- state3  reads=['B'] writes=['out']  work=1 depth=1
"""

#: The agent's view after it expands ``state0``: that state's kernels as rows, their bodies still closed.
OPENED = """\
SDFG 'shaped'
|- state0  reads=['A', 'B'] writes=['B', 's']  work=21 depth=2
|  `- doall0  [i=0:20]  reads=['A'] writes=['B']  work=20 depth=1
`- for0  i=1:20  num_states=3  work=19 depth=19
   |- state1
   `- if0
      |- block0  when A[i] > 0.0
      |  `- state2  reads=['B', 's'] writes=['out']  work=1 depth=1
      `- block1  else
         `- state3  reads=['B'] writes=['out']  work=1 depth=1
"""

#: After it also expands ``doall0``: the doall's statements, which fission splits.
OPENED_DOALL = """\
SDFG 'shaped'
|- state0  reads=['A', 'B'] writes=['B', 's']  work=21 depth=2
|  `- doall0  [i=0:20]  reads=['A'] writes=['B']  work=20 depth=1
|     : tmp3 = A[i]
|     : tmp0 = tmp3 + 1.0
|     : B[i] = tmp0
`- for0  i=1:20  num_states=3  work=19 depth=19
   |- state1
   `- if0
      |- block0  when A[i] > 0.0
      |  `- state2  reads=['B', 's'] writes=['out']  work=1 depth=1
      `- block1  else
         `- state3  reads=['B'] writes=['out']  work=1 depth=1
"""

#: What an agent sees before it expands anything: every block, no state's kernels.
COLLAPSED = """\
SDFG 'shaped'
|- state0  doalls=1  reads=['A', 'B'] writes=['B', 's']  work=21 depth=2
`- for0  i=1:20  num_states=3  work=19 depth=19
   |- state1
   `- if0
      |- block0  when A[i] > 0.0
      |  `- state2  reads=['B', 's'] writes=['out']  work=1 depth=1
      `- block1  else
         `- state3  reads=['B'] writes=['out']  work=1 depth=1
"""


def tree_of(program: DaceProgram) -> str:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = "shaped"
    normalize_labels(sdfg)
    return describe_graph(sdfg)


def test_the_tree_format_is_pinned() -> None:
    """If this fails, the format changed. Update GOLDEN only when that change is the intent -- every
    agent prompt reads this shape."""
    assert tree_of(shaped) + "\n" == GOLDEN


def test_a_state_shows_its_kernels_only_once_expanded() -> None:
    sdfg = shaped.to_sdfg(simplify=True)
    sdfg.name = "shaped"
    normalize_labels(sdfg)

    collapsed = describe_graph(sdfg, expanded=())
    opened = describe_graph(sdfg, expanded=["state0"])
    opened_doall = describe_graph(sdfg, expanded=["state0", "doall0"])

    assert collapsed + "\n" == COLLAPSED
    assert opened + "\n" == OPENED
    assert opened_doall + "\n" == OPENED_DOALL


def test_every_line_is_a_guide_then_one_labelled_thing() -> None:
    for line in tree_of(shaped).splitlines()[1:]:
        assert re.match(
            r"^(\|  |   )*(\|- |`- )"
            r"(state|for|if|block|continue|break|return|doall)\d+\b",
            line,
        ), line


def test_every_row_the_session_prints_is_a_label_its_labeled_calls_take() -> None:
    """Reading the tree and acting on it use one vocabulary: each row's first word is a key of the row index."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    session = Session(sdfg)
    labels = [re.sub(r"^[|` -]*", "", line).split()[0] for line in session.describe().splitlines()[1:]]
    assert labels and all(label in session.row_index() for label in labels), labels


def test_a_collapsed_row_hides_what_its_expand_showed_and_a_row_not_open_says_so() -> None:
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    sut = Session(sdfg)
    closed = sut.describe()
    state = next(label for label, (obj, _) in sut.row_index().items() if isinstance(obj, dc.SDFGState))
    assert sut.view(expand=[state]) != f"ok: expanded {state}\n{closed}", "expanding shows the state's rows"

    answer = sut.view(collapse=[state])

    assert answer == f"ok: collapsed {state}\n{closed}"
    assert sut.view(collapse=[state]).startswith(f"not open: {state};")


@dc.program
def branchy(A: dc.float64[20], B: dc.float64[20]) -> None:
    for i in dc.map[0:20]:
        if A[i] > 0.0:
            B[i] = A[i]
        else:
            B[i] = -A[i]


def nested_state_labels(session: Session) -> list[str]:
    """The labels of the states inside the doall bodies."""
    return [
        label
        for label, (obj, _) in session.row_index().items()
        if isinstance(obj, dc.SDFGState) and obj.sdfg is not session.sdfg
    ]


def shown_labels(tree: str) -> set[str]:
    return {re.sub(r"^[|` -]*", "", line).split()[0] for line in tree.splitlines()[1:]}


def test_an_expanded_state_shows_its_doall_and_the_doall_opens_its_body_states_alone() -> None:
    sut = Session(branchy.to_sdfg(simplify=True))
    (doall,) = [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, dc.nodes.MapEntry)]
    (state,) = [label for label, (obj, _) in sut.row_index().items() if obj is sut.row_index()[doall][1]]
    inner = nested_state_labels(sut)
    assert inner and not shown_labels(sut.describe()) & set(inner), "a closed state hides the doall's body states"

    state_open = sut.view(expand=[state])
    opened = sut.view(expand=[doall])
    folded = sut.view(collapse=[doall])

    assert doall in shown_labels(state_open) and not shown_labels(state_open) & set(inner), "the doall stays closed"
    assert set(inner) <= shown_labels(opened), "expanding the doall shows its body states"
    assert folded.startswith(f"ok: collapsed {doall}\n")
    assert doall in shown_labels(folded) and not shown_labels(folded) & set(inner)
    assert sut.view(expand=[doall], collapse=[doall]).endswith(folded.partition("\n")[2]), "one call opens, then closes"


# conditions read as the arrays they test


def test_a_condition_names_the_array_it_really_reads() -> None:
    """The frontend hoists the scalar read to an interstate assignment (`A_index = A[1 + i]`) and the
    branch then tests a name that means nothing on its own."""
    tree = tree_of(shaped)
    assert "when A[i] > 0.0" in tree, tree
    assert "A_index" not in tree


def test_a_name_with_two_definitions_is_left_alone() -> None:
    """Which definition reaches a block depends on the path taken, so folding either one in would show
    a condition the program does not always evaluate."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_labels(sdfg)
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


def test_resolution_terminates_on_a_self_referential_definition() -> None:
    """`i = i + 1` on a back edge is ordinary; substituting it forever is not."""
    assert resolve_scalars("i < N", {"i": "i + 1"}) == "i + 1 < N"


def test_resolution_follows_a_chain_to_the_array() -> None:
    assert resolve_scalars("c > 0", {"c": "b * 2", "b": "A[k]"}) == "A[k] * 2 > 0"


def test_an_unparsable_condition_is_passed_through() -> None:
    assert resolve_scalars("this is not python", {}) == "this is not python"


# bodies: what each kernel computes


@dc.program
def nested_maps(A: dc.float64[8, 4], B: dc.float64[8, 4]) -> None:
    """An outer kernel whose body is another kernel."""
    for i in dc.map[0:8]:
        for j in dc.map[0:4]:
            B[i, j] = A[i, j] * 2.0


def with_bodies(program: DaceProgram) -> str:
    sdfg = program.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    return describe_graph(sdfg, bodies=True)


def test_an_index_is_simplified_even_when_nothing_is_defined() -> None:
    """Whether a program assigns anything elsewhere must not change how one condition prints."""
    assert resolve_scalars("A[(1 + (1 * i))] > 0.0", {}) == "A[i + 1] > 0.0"


def test_a_nested_sdfg_assignment_does_not_define_an_outer_name() -> None:
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
def test_a_loop_range_ends_one_past_its_last_value_in_its_direction(
    init: str, condition: str, update: str, want: str
) -> None:
    """A descending loop printed with ``end + 1`` reads as stopping two values early."""
    from dace.sdfg.state import LoopRegion

    loop = LoopRegion("loop", condition, "i", init, update)

    assert introspect.loop_domain(loop, {}) == want


def test_bodies_are_off_by_default() -> None:
    """An emit per kernel is not free, and the structure alone is what a fusion decision needs."""
    assert not [line for line in tree_of(shaped).splitlines() if introspect.BODY in line]
    assert [line for line in with_bodies(shaped).splitlines() if introspect.BODY in line], "nothing to be off"


def test_a_leaf_kernel_prints_what_it_computes() -> None:
    tree = with_bodies(shaped)
    assert f"{introspect.BODY}B[i] = " in tree, tree
    for line in tree.splitlines():
        if introspect.BODY in line:
            assert line.split(introspect.BODY, 1)[0].strip("| `") == "", "a body line must sit under its kernel"


def test_a_body_does_not_repeat_its_headers() -> None:
    """The kernel line already shows the domain those `for` headers iterate."""
    for line in with_bodies(shaped).splitlines():
        if introspect.BODY in line:
            assert not line.split(introspect.BODY, 1)[1].startswith("for "), line


def test_a_kernel_containing_a_kernel_has_no_body_of_its_own() -> None:
    """`map_body_lines` recurses into a nested map, and the tree already gives that map its own row,
    so emitting at both levels would print the inner kernel twice."""
    sdfg = nested_maps.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    nested = [
        (st, n)
        for st in sdfg.states()
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
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry) and st.entry_node(n) is not None
    ]
    assert any(kernel_body(st, sdfg, n, st.scope_children()) for st, n in inner), "inner printed nothing"


def test_an_emitter_refusal_is_reported_on_the_line_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tree is a read-only view. A nest the Python projection cannot express is exactly what the
    agent needs to be told about, so it must not take the whole tree down."""

    def refuse(state: object, entry: object, body_only: bool = False) -> Never:
        raise UnsupportedNest("no emitter for this")

    monkeypatch.setattr(introspect, "map_python", refuse)
    tree = with_bodies(shaped)
    assert "<not emitted: no emitter for this>" in tree


# reductions on the kernel line


@dc.program
def matvec(A: dc.float64[8, 4], B: dc.float64[4], C: dc.float64[8]) -> None:
    """A WCR over one of two map axes -- the classic tree reduction."""
    for i, j in dc.map[0:8, 0:4]:
        C[i] += A[i, j] * B[j]


def test_a_reduction_is_named_on_the_kernel_line() -> None:
    """A WCR on a map IS a tree reduction: the map declares its iterations independent, so the fold
    order is unspecified and a backend may use a register accumulator or an OpenMP clause. That is
    structural, and the agent should not have to read the body to find it."""
    tree = tree_of(matvec)
    assert "accum=(C += over j)" in tree, tree


def test_only_the_collapsed_axis_is_reported() -> None:
    """The reduced axes are the map parameters the output subset does not mention -- a map over
    (i, j) writing C[i] has collapsed j and only j."""
    tree = tree_of(matvec)
    assert "C += over j)" in tree and "over i" not in tree, tree


def test_a_kernel_without_a_reduction_says_nothing() -> None:
    assert "reduce=" not in tree_of(shaped)


def test_the_reduction_op_is_read_off_the_wcr() -> None:
    sdfg = matvec.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    state, entry = next((st, n) for st in sdfg.states() for n in st.nodes() if isinstance(n, dc.sdfg.nodes.MapEntry))
    assert introspect.kernel_reductions(state, entry) == ["C += over j"]
    # A different op reads as itself, not as "+".
    exit_node = state.exit_node(entry)
    edge = next(e for e in state.in_edges(exit_node) if e.data.wcr is not None)
    edge.data.wcr = "lambda x, y: max(x, y)"
    assert introspect.kernel_reductions(state, entry) == ["C max= over j"]


def test_a_body_is_not_recovered_by_slicing_the_emitted_block() -> None:
    """The body is emitted as one iteration's statements, not recovered by dropping len(params) lines off the
    emitted loop block and dedenting by 4 * len(params)."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    state, entry = next(
        (st, n)
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry) and st.entry_node(n) is None
    )
    body = kernel_body(state, sdfg, entry, state.scope_children())
    assert body, "the fixture kernel emits nothing"
    for line in body:
        assert not line.startswith(" "), f"a body line arrived still indented: {line!r}"
        assert not line.startswith("for "), f"a header leaked into the body: {line!r}"
    # and it agrees with the full block the emitter produces for the same kernel
    full = introspect.kernel_source(state, sdfg, entry)
    assert all(f"    {line}\n" in full for line in body), (body, full)


# one kernel's body


def first_nest(program: DaceProgram) -> tuple[dc.SDFG, SDFGState, dc.nodes.MapEntry]:
    sdfg = program.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    state = next(st for st in sdfg.states() if any(isinstance(n, dc.nodes.MapEntry) for n in st.nodes()))
    entry = next(n for n in state.scope_children()[None] if isinstance(n, dc.nodes.MapEntry))
    return sdfg, state, entry


def test_a_kernel_body_is_the_statements_without_their_headers() -> None:
    sdfg, state, entry = first_nest(shaped)
    body = kernel_body(state, sdfg, entry, state.scope_children())
    assert body and all(isinstance(line, str) for line in body)
    assert not any(line.startswith("for ") for line in body), "headers are on the kernel line already"


def test_a_reduction_body_is_folded() -> None:
    """An explicit accumulate is the only point rendering of a reduction; `np.sum` is a whole-array
    spelling that belongs to the slice form."""
    sdfg, state, entry = first_nest(matvec)
    body = "\n".join(kernel_body(state, sdfg, entry, state.scope_children()))
    assert "C[i] = C[i] +" in body, body
    assert "np.sum" not in body


# a kernel's representation: pure, runnable numpy


def source_of_first_kernel(program: DaceProgram) -> tuple[dc.SDFG, SDFGState, str]:
    sdfg, state, entry = first_nest(program)
    return sdfg, state, introspect.kernel_source(state, sdfg, entry)


def test_a_kernel_source_is_a_whole_module_not_a_fragment() -> None:
    """`kernel_body` is the excerpt the tree prints; its statements reference loop variables that only
    exist inside their headers. The representation has to be something an agent can run."""
    _, _, source = source_of_first_kernel(shaped)
    assert source.startswith("import numpy as np")
    assert "\ndef doall" in source
    compile(source, "<kernel>", "exec")  # syntactically a module, not a snippet


def test_a_kernel_source_runs_with_nothing_injected() -> None:
    """No `np` handed in -- the source imports what it calls or it does not count."""
    _, _, source = source_of_first_kernel(matvec)
    namespace: dict[str, Any] = {}  # what exec defines
    exec(source, namespace)  # a bare dict: only what the source itself defines
    assert "np" in namespace and any(k.startswith("doall") and callable(v) for k, v in namespace.items())


def test_a_kernel_source_computes_what_the_sdfg_computes() -> None:
    """The point of it being runnable: emit, execute, compare. A representation nothing can execute
    cannot be shown to be correct."""
    sdfg, _, source = source_of_first_kernel(matvec)
    namespace: dict[str, Any] = {}  # what exec defines
    exec(source, namespace)
    kernel = next(v for k, v in namespace.items() if k.startswith("doall") and callable(v))

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
def last_column(A: dc.float64[N, M], B: dc.float64[N]) -> None:
    for i in dc.map[0:N]:
        B[i] = A[i, M - 1]


def test_a_kernel_source_takes_a_symbol_only_its_body_reads() -> None:
    """``M`` bounds no map, yet the body indexes with it, so the signature must take it."""
    _, _, source = source_of_first_kernel(last_column)
    namespace: dict[str, Any] = {}  # what exec defines
    exec(source, namespace)
    kernel = next(v for k, v in namespace.items() if k.startswith("doall") and callable(v))
    A, B = np.arange(12, dtype=np.float64).reshape(3, 4).copy(), np.zeros(3)

    kernel(A=A, B=B, M=4, N=3)

    np.testing.assert_array_equal(B, A[:, 3])


def test_only_states_and_maps_expand() -> None:
    session = Session(shaped.to_sdfg(simplify=True))
    loop = next(label for label, (obj, _) in session.row_index().items() if isinstance(obj, LoopRegion))

    sut = session.expand([loop])

    assert sut == f"illegal: only states and maps expand; {loop} is neither."
    assert session.expanded == {}


@dc.program
def scaled_or_copied(x: dc.float64[3], y: dc.float64[N]) -> None:
    if x[0] > 0.0:
        for i in dc.map[0:N]:
            y[i] = 2 * x[1]
    else:
        for i in dc.map[0:N]:
            y[i] = x[2]


@dc.program
def calls_with_a_row(rows: dc.float64[N, 3], out: dc.float64[N], k: dc.int64) -> None:
    row = rows[k]
    scaled_or_copied(row, out)


def test_a_callee_given_a_row_of_an_array_shows_its_branches_and_doalls_in_the_tree(tmp_path: Path) -> None:
    """The row is a slice View, behind which DaCe's inlining left the callee nested at the top of its state: the tree
    then showed one empty state and offered no move (qe_vexx_k's g2_convolution)."""
    session = Session(calls_with_a_row.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 8})

    tree = session.canonicalize()

    assert top_level_nsdfgs(session.sdfg) == []
    assert "when rows[k, 0] > 0.0" in tree and tree.count("doalls=1") == 2
    assert {move.kind for move in session.list_moves()} >= {"tile-doall"}


@dc.program
def until_small(x: dc.float64[1], y: dc.float64[20]) -> None:
    while x[0] > 1.0:
        x[0] = x[0] / 2.0
    for i in dc.map[0:20]:
        y[i] = y[i] * x[0]


@dc.program
def blur_scale(A: dc.float64[20], T: dc.float64[20], B: dc.float64[20]) -> None:
    for i in dc.map[1:19]:
        T[i] = (A[i - 1] + A[i] + A[i + 1]) / 3
    for i in dc.map[1:19]:
        B[i] = 2 * T[i]


def test_a_loop_that_runs_until_its_condition_fails_is_a_while_row_and_a_counted_one_a_for_row() -> None:
    sdfg = until_small.to_sdfg(simplify=True)
    normalize_labels(sdfg)

    tree = describe_graph(sdfg, expanded=())

    assert re.search(r"while\d+  x_index > 1\.0", tree), tree
    assert not re.search(r"\bfor\d+", tree), tree


def test_generated_temporaries_print_as_tmp_and_arguments_keep_their_names(tmp_path: Path) -> None:
    sut = Session(blur_scale.to_sdfg(simplify=True), work_dir=str(tmp_path))
    sut.canonicalize()

    tree = sut.view(expand=[label for label in sut.row_index() if label.startswith(("state", "doall"))])
    names = {name for sd in sut.sdfg.all_sdfgs_recursive() for name in sd.arrays}

    assert {"A", "T", "B"} <= names
    assert not [name for name in names if "_slice" in name or name.startswith("_")], names
    assert re.search(r": tmp\d+ = A\[_loop_it_0 - 1\] \+ A\[_loop_it_0\]", tree), tree


def test_after_a_fusion_no_rename_line_is_printed_and_no_name_means_two_containers(tmp_path: Path) -> None:
    sut = Session(blur_scale.to_sdfg(simplify=True), work_dir=str(tmp_path))
    sut.canonicalize()
    (fusion,) = sut.list_moves("doall-fusion")
    sut.apply_move(fusion.kind, fusion.labels)
    opened = [label for label in sut.row_index() if label.startswith(("state", "doall"))]

    python = sut.view(expand=opened)
    c = sut.view(c=[label for label in opened if label.startswith("state")])

    assert "_loop_it_1 = _loop_it_0" not in python, python
    stores = [line for line in c.splitlines() if re.search(r": T\[", line)]
    assert len(stores) == 1 and re.search(r"T\[_loop_it_0\] = tmp\d+\[0\];", stores[0]), c


def test_a_generated_local_shortens_to_v_and_an_index_read_to_its_array() -> None:
    lines = ["prog_A_index = A[i]", "B_slice = prog_A_index + 1.0", "B[i] = B_slice"]

    shortened = introspect.shortened_locals(lines, frozenset({"A", "B", "i"}))

    assert shortened == ["A_index = A[i]", "v0 = A_index + 1.0", "B[i] = v0"]


def test_a_symbol_renamed_once_is_folded_and_a_container_copy_stays() -> None:
    lines = ["t = a[k]", "j = k", "b[j] = t", "c = t"]

    folded = introspect.folded_renames(lines, frozenset({"a", "b", "c", "t"}))

    assert folded == ["t = a[k]", "b[k] = t", "c = t"]
