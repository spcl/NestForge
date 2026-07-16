"""Splitting the computation around an unsupported (non-emittable) library node into distinct states, so
the externalize lane can offload the pure compute before/after it while the node itself stays native
(:mod:`nestforge.split_unsupported`). An MPI ``Bcast`` stands in for the general unsupported node.
"""
import importlib

import dace
import pytest
from dace.sdfg import nodes

from nestforge.emit_libnode import UnsupportedLibraryNode, emit_library_node, is_emittable_library_node
from nestforge.split_unsupported import (isolate_into_own_state, isolate_unsupported_library_nodes,
                                         unsupported_library_nodes)

N = dace.symbol("N")

#: (module, class) for a spread of MPI library nodes -- collectives, point-to-point, and completion. Every
#: one must be refused as a communication node. ``reduce`` is the name-collision case: an MPI ``Reduce``
#: shares its class name with the registered standard ``Reduce``.
MPI_NODES = [
    ("bcast", "Bcast"),
    ("allreduce", "Allreduce"),
    ("isend", "Isend"),
    ("irecv", "Irecv"),
    ("send", "Send"),
    ("recv", "Recv"),
    ("allgather", "Allgather"),
    ("gather", "Gather"),
    ("scatter", "Scatter"),
    ("reduce", "Reduce"),
    ("wait", "Wait"),
]


def make_mpi_node(module, cls):
    node_cls = vars(importlib.import_module(f"dace.libraries.mpi.nodes.{module}"))[cls]
    return node_cls(cls.lower())


def linear_chain_sdfg():
    """A single state with A -> t1 -> tmp -> M -> tmp2 -> t3 -> B (M a plain tasklet stand-in)."""
    sdfg = dace.SDFG("chain")
    for a in ("A", "B"):
        sdfg.add_array(a, [N], dace.float64)
    for a in ("tmp", "tmp2"):
        sdfg.add_transient(a, [N], dace.float64)
    st = sdfg.add_state("main")
    tmp, tmp2 = st.add_access("tmp"), st.add_access("tmp2")
    t1 = st.add_tasklet("t1", {"i0"}, {"o0"}, "o0 = i0 + 1.0")
    m = st.add_tasklet("M", {"i0"}, {"o0"}, "o0 = i0 * 2.0")
    t3 = st.add_tasklet("t3", {"i0"}, {"o0"}, "o0 = i0 - 3.0")
    st.add_edge(st.add_read("A"), None, t1, "i0", dace.Memlet("A[0]"))
    st.add_edge(t1, "o0", tmp, None, dace.Memlet("tmp[0]"))
    st.add_edge(tmp, None, m, "i0", dace.Memlet("tmp[0]"))
    st.add_edge(m, "o0", tmp2, None, dace.Memlet("tmp2[0]"))
    st.add_edge(tmp2, None, t3, "i0", dace.Memlet("tmp2[0]"))
    st.add_edge(t3, "o0", st.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, st, m


def test_isolate_into_own_state_produces_valid_three_way_split():
    sdfg, st, m = linear_chain_sdfg()
    isolate_into_own_state(sdfg, st, m)
    states = list(sdfg.states())
    assert len(states) == 3
    # the node's state holds only the node (plus its in/out access nodes) -- no other tasklet.
    m_states = [s for s in states if m in s.nodes()]
    assert len(m_states) == 1
    others = [n for n in m_states[0].nodes() if n is not m and not isinstance(n, nodes.AccessNode)]
    assert others == []
    sdfg.validate()  # the split must leave a valid SDFG


def bcast_sdfg():
    """A single state mixing a producer tasklet, an MPI ``Bcast`` (unsupported), and a consumer tasklet."""
    from dace.libraries.mpi.nodes.bcast import Bcast
    sdfg = dace.SDFG("with_bcast")
    for a in ("A", "B"):
        sdfg.add_array(a, [8], dace.float64)
    for a in ("tmp", "tmp2"):
        sdfg.add_transient(a, [8], dace.float64)
    sdfg.add_array("root", [1], dace.int32)
    st = sdfg.add_state("main")
    tmp, tmp2 = st.add_access("tmp"), st.add_access("tmp2")
    prod = st.add_tasklet("prod", {"i0"}, {"o0"}, "o0 = i0 + 1.0")
    cons = st.add_tasklet("cons", {"i0"}, {"o0"}, "o0 = i0 - 1.0")
    bcast = Bcast("bcast")
    st.add_edge(st.add_read("A"), None, prod, "i0", dace.Memlet("A[0]"))
    st.add_edge(prod, "o0", tmp, None, dace.Memlet("tmp[0]"))
    st.add_edge(tmp, None, bcast, "_inbuffer", dace.Memlet("tmp[0:8]"))
    st.add_edge(st.add_read("root"), None, bcast, "_root", dace.Memlet("root[0]"))
    st.add_edge(bcast, "_outbuffer", tmp2, None, dace.Memlet("tmp2[0:8]"))
    st.add_edge(tmp2, None, cons, "i0", dace.Memlet("tmp2[0]"))
    st.add_edge(cons, "o0", st.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, st, bcast


def test_unsupported_predicate_flags_mpi_not_registered_nodes():
    sdfg, st, bcast = bcast_sdfg()
    assert bcast in unsupported_library_nodes(st)
    # a registered node (MatMul) is NOT flagged.
    from dace.libraries.blas.nodes.matmul import MatMul
    st.add_node(MatMul("mm"))
    flagged = unsupported_library_nodes(st)
    assert bcast in flagged and all(type(n).__name__ != "MatMul" for n in flagged)


def test_pass_isolates_bcast_and_is_idempotent():
    sdfg, st, bcast = bcast_sdfg()
    n_isolated = isolate_unsupported_library_nodes(sdfg)
    assert n_isolated == 1
    states = list(sdfg.states())
    assert len(states) == 3  # producers | bcast island | consumers
    island = [s for s in states if bcast in s.nodes()]
    assert len(island) == 1
    # the Bcast is alone with its access nodes -- the producer/consumer tasklets are elsewhere.
    non_access = [n for n in island[0].nodes() if not isinstance(n, nodes.AccessNode)]
    assert non_access == [bcast]
    # every other state is free of unsupported nodes.
    assert all(not unsupported_library_nodes(s) for s in states if s is not island[0])
    # idempotent: a second run finds nothing to split.
    assert isolate_unsupported_library_nodes(sdfg) == 0


@pytest.mark.parametrize("module,cls", MPI_NODES)
def test_every_mpi_node_is_refused_and_flagged_unsupported(module, cls):
    node = make_mpi_node(module, cls)
    assert not is_emittable_library_node(node)
    sdfg = dace.SDFG(f"only_{cls.lower()}")
    st = sdfg.add_state()
    st.add_node(node)
    assert node in unsupported_library_nodes(st)
    with pytest.raises(UnsupportedLibraryNode, match="communication"):
        emit_library_node(node, st, sdfg)


def test_mpi_reduce_name_collision_does_not_route_to_standard_reduce():
    # MPI Reduce and the registered standard Reduce share the class name "Reduce"; the module check must
    # win so the MPI one is refused rather than emitted as a standard reduction.
    mpi_reduce = make_mpi_node("reduce", "Reduce")
    from dace.libraries.standard.nodes.reduce import Reduce as StdReduce
    std_reduce = StdReduce("std", "lambda a, b: a + b")
    assert not is_emittable_library_node(mpi_reduce)
    assert is_emittable_library_node(std_reduce)


def two_bcast_chain_sdfg():
    """A -> prod -> t0 -> bcast1 -> t1 -> mid -> t2 -> bcast2 -> t3 -> cons -> B (two MPI nodes, one state)."""
    from dace.libraries.mpi.nodes.bcast import Bcast
    sdfg = dace.SDFG("two_bcast")
    for a in ("A", "B"):
        sdfg.add_array(a, [8], dace.float64)
    for a in ("t0", "t1", "t2", "t3"):
        sdfg.add_transient(a, [8], dace.float64)
    sdfg.add_array("root", [1], dace.int32)
    st = sdfg.add_state("main")
    t0, t1, t2, t3 = (st.add_access(a) for a in ("t0", "t1", "t2", "t3"))
    root = st.add_read("root")
    prod = st.add_tasklet("prod", {"i0"}, {"o0"}, "o0 = i0 + 1.0")
    mid = st.add_tasklet("mid", {"i0"}, {"o0"}, "o0 = i0 * 2.0")
    cons = st.add_tasklet("cons", {"i0"}, {"o0"}, "o0 = i0 - 1.0")
    b1, b2 = Bcast("bcast1"), Bcast("bcast2")
    st.add_edge(st.add_read("A"), None, prod, "i0", dace.Memlet("A[0]"))
    st.add_edge(prod, "o0", t0, None, dace.Memlet("t0[0]"))
    st.add_edge(t0, None, b1, "_inbuffer", dace.Memlet("t0[0:8]"))
    st.add_edge(root, None, b1, "_root", dace.Memlet("root[0]"))
    st.add_edge(b1, "_outbuffer", t1, None, dace.Memlet("t1[0:8]"))
    st.add_edge(t1, None, mid, "i0", dace.Memlet("t1[0]"))
    st.add_edge(mid, "o0", t2, None, dace.Memlet("t2[0]"))
    st.add_edge(t2, None, b2, "_inbuffer", dace.Memlet("t2[0:8]"))
    st.add_edge(root, None, b2, "_root", dace.Memlet("root[0]"))
    st.add_edge(b2, "_outbuffer", t3, None, dace.Memlet("t3[0:8]"))
    st.add_edge(t3, None, cons, "i0", dace.Memlet("t3[0]"))
    st.add_edge(cons, "o0", st.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, [b1, b2]


def test_two_mpi_nodes_in_sequence_are_each_isolated():
    sdfg, bcasts = two_bcast_chain_sdfg()
    n_isolated = isolate_unsupported_library_nodes(sdfg)
    assert n_isolated == 2
    states = list(sdfg.states())
    # every MPI node is alone in its own state (only the node + its access nodes).
    for b in bcasts:
        island = [s for s in states if b in s.nodes()]
        assert len(island) == 1
        assert [n for n in island[0].nodes() if not isinstance(n, nodes.AccessNode)] == [b]
    # no state still mixes an unsupported node with compute.
    assert isolate_unsupported_library_nodes(sdfg) == 0


def test_independent_compute_alongside_mpi_is_preserved():
    # A state with the bcast chain PLUS an independent tasklet branch (C -> ind -> D): isolating the bcast
    # must not drop the independent branch -- it lands in a surrounding state, still present.
    sdfg, st, bcast = bcast_sdfg()
    sdfg.add_array("C", [8], dace.float64)
    sdfg.add_array("D", [8], dace.float64)
    ind = st.add_tasklet("ind", {"i0"}, {"o0"}, "o0 = i0 + 5.0")
    st.add_edge(st.add_read("C"), None, ind, "i0", dace.Memlet("C[0]"))
    st.add_edge(ind, "o0", st.add_write("D"), None, dace.Memlet("D[0]"))
    assert isolate_unsupported_library_nodes(sdfg) == 1
    all_tasklets = [n.label for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.Tasklet)]
    assert "ind" in all_tasklets  # the independent branch survived the split
    assert isolate_unsupported_library_nodes(sdfg) == 0
