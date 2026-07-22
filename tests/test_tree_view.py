"""The ASCII tree the agent reads (:func:`nestforge.introspect.describe_graph`).

The tree is the agent's whole view of the program, so its FORMAT is an interface: a change to it
changes what every agent prompt sees. The golden test below pins it in full, so a format change has
to be made deliberately rather than drifting out of an unrelated edit.
"""
import re

import pytest

pytest.importorskip("dace")

import dace as dc

from nestforge.introspect import describe_graph, interstate_definitions, resolve_scalars
from nestforge.normalize import normalize_for_tree
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
        assert re.match(r"^(\|  |   )*(\|- |`- )"
                        r"(state|for|while|if|block|continue|break|return|kernel)\d+_\d+\b", line), line


def test_indentation_tracks_the_level_in_the_label():
    """A line's depth in the tree and the level in its canonical name are the same number, so the
    agent can read either one."""
    for line in tree_of(shaped).splitlines()[1:]:
        guide, body = re.match(r"^((?:\|  |   )*(?:\|- |`- ))(.*)$", line).groups()
        level = int(re.match(r"^[a-z]+(\d+)_", body).group(1))
        assert len(guide) == 3 * (level + 1), f"level {level} at guide width {len(guide)}: {line}"


def test_session_stamps_the_ids_that_act_on_each_line():
    """B1: reading the tree and acting on it use ONE vocabulary. A nest line carries the very handle
    can_fuse/fuse resolve -- not a label the agent has to match against a separate list_nests call."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    tree = session.describe()
    nest_ids = re.findall(r"\[(e\d+:nest:\d+)\]", tree)
    assert nest_ids, "no nest line carried a minted handle"
    for hid in nest_ids:
        assert session.resolve(hid, "nest") is not None
    # Two nests off the tree are exactly what can_fuse accepts -- no list_nests call in between.
    assert isinstance(session.can_fuse(nest_ids[0], nest_ids[1]), str)


def test_region_lines_carry_the_stable_descriptive_id_not_a_minted_one():
    """Nothing resolves a ``region`` kind, so minting one would grow the registry on a read-only call
    and hand back an id that raises on the kind guard."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    tree = session.describe()
    assert re.search(r"\[region:state0_0\]", tree), tree
    assert not re.search(r"\[e\d+:region:", tree)


# --- conditions read as the arrays they test -------------------------------------------------------


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
    edge = next(e for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges()
                if "A_index" in e.data.assignments)
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
