# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An empty memlet between two maps orders them and reads nothing, so it gives ``doall-fusion`` no intermediate.

``efficientnet_b1`` and ``aes_graupel`` hold such an edge from a producer's output to a consumer map; the fusion check
then handed its ``None`` subset to ``OTFMapFusion``, whose ``AttributeError`` broke the whole move menu.
"""

import dace
from dace.sdfg import nodes

from nestforge.stages.moves import intermediates, legal_moves

#: Elements of each array.
SIZE = 16
#: Iterations of the consumer map: unequal to the producer's, so ``MapFusionVertical`` refuses and the check reaches
#: ``OTFMapFusion``, as on efficientnet_b1.
CONSUMED = 8


def ordered_by_an_empty_memlet(reads_t: bool) -> tuple[dace.SDFG, nodes.MapEntry, nodes.MapEntry]:
    """``T = A + 1``, then ``B[:8] = 2 A[:8]`` (or ``2 T[:8]`` when ``reads_t``) ordered after ``T`` by an empty
    memlet."""
    sdfg = dace.SDFG(f"empty_memlet_{reads_t}")
    for name in ("A", "B"):
        sdfg.add_array(name, [SIZE], dace.float64)
    sdfg.add_transient("T", [SIZE], dace.float64)
    state = sdfg.add_state()
    a, t = state.add_read("A"), state.add_access("T")
    _, produce, _ = state.add_mapped_tasklet(
        "produce",
        {"i": f"0:{SIZE}"},
        {"a": dace.Memlet("A[i]")},
        "o = a + 1.0",
        {"o": dace.Memlet("T[i]")},
        external_edges=True,
        input_nodes={"A": a},
        output_nodes={"T": t},
    )
    source = "T" if reads_t else "A"
    _, consume, _ = state.add_mapped_tasklet(
        "consume",
        {"j": f"0:{CONSUMED}"},
        {"x": dace.Memlet(f"{source}[j]")},
        "y = 2.0 * x",
        {"y": dace.Memlet("B[j]")},
        external_edges=True,
        input_nodes={source: t if reads_t else a},
        output_nodes={"B": state.add_write("B")},
    )
    state.add_nedge(t, consume, dace.Memlet())
    sdfg.validate()
    return sdfg, produce, consume


def test_an_empty_memlet_is_no_intermediate() -> None:
    sdfg, produce, consume = ordered_by_an_empty_memlet(reads_t=False)
    state = sdfg.start_state
    assert intermediates(state, produce, consume) == []
    # listing raised AttributeError before; the two maps share no data, so only a horizontal fusion may be offered
    assert all(kind == "doall-fusion" for kind, _ in legal_moves(sdfg, "doall-fusion"))


def test_a_read_beside_an_empty_memlet_is_still_an_intermediate() -> None:
    sdfg, produce, consume = ordered_by_an_empty_memlet(reads_t=True)
    state = sdfg.start_state
    assert [a.data for a in intermediates(state, produce, consume)] == ["T"]
