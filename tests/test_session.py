"""Session: the epoch-stamped-id safety layer over the 4-phase API, and the three distinct decision axes
it exposes -- region structure (Level 1), nest fusion (Level 2), and offload (Phase 2). These tests cover
the layer Session ADDS -- id minting, the stale-handle guard on every mutation, kind-checking, the
region/nest distinction, and that each tool returns plain JSON-able data (never a live node). The wrapped
transforms have their own tests; here we only prove Session drives them safely.
"""
import numpy as np
import pytest
import dace

from nestforge.session import Session, StaleHandle

N = dace.symbol('N')


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient -> the two maps fuse vertically
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def two_indep(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = B[i] * 3.0


def make_session():
    return Session(vertical_pair.to_sdfg(simplify=True))


def barred_session():
    # simplify=False keeps each map in its own state -> a state barrier between the two nests.
    return Session(two_indep.to_sdfg(simplify=False))


# --- Level 2: nest fusion + the id/epoch safety layer ---------------------------------------------


def test_list_nests_is_plain_data():
    s = make_session()
    assert "[fusion barrier]" in s.describe()
    nests = s.list_nests()
    assert len(nests) == 2
    assert {n["id"] for n in nests} == {"e0:nest:0", "e0:nest:1"}  # epoch-0 stamped ids
    assert ({"A", "B"}, {"T"}) in [(set(n["reads"]), set(n["writes"])) for n in nests]
    for n in nests:  # JSON-able: only str/bool/list, never a node
        assert isinstance(n["label"], str) and isinstance(n["parallel"], bool)


def test_can_fuse_uses_nest_ids():
    s = make_session()
    a, b = (n["id"] for n in s.list_nests())
    assert s.can_fuse(a, b) == "yes"


def test_fuse_bumps_epoch_and_stales_prior_ids():
    s = make_session()
    nests = s.list_nests()
    moves = s.list_fusions()
    assert len(moves) == 1 and moves[0]["kind"] == "fuse-map-vertical"
    assert s.epoch == 0
    s.fuse(moves[0]["id"])
    assert s.epoch == 1
    with pytest.raises(StaleHandle):  # a nest id from epoch 0 no longer resolves
        s.can_fuse(nests[0]["id"], nests[1]["id"])


def test_fission_all_bumps_epoch():
    s = make_session()
    s.list_fusions()  # mint some epoch-0 handles
    s.fission_all()
    assert s.epoch == 1
    assert s.handles == {}  # all handles dropped on the bump


def test_resolve_rejects_wrong_kind():
    s = make_session()
    nest_id = s.list_nests()[0]["id"]
    with pytest.raises(KeyError):  # a nest id is not a move id
        s.fuse(nest_id)


def test_unknown_id_at_current_epoch_is_not_stale():
    s = make_session()
    s.list_nests()
    with pytest.raises(KeyError) as ei:  # current-epoch but nonexistent -> plain unknown, not StaleHandle
        s.can_fuse("e0:nest:99", "e0:nest:0")
    assert not isinstance(ei.value, StaleHandle)


# --- Level 1: region structure (containers) + the merge-first ordering rule -----------------------


def test_region_tree_exposes_containers_and_their_nests():
    s = make_session()
    tree = s.region_tree()
    assert tree["type"] == "SDFG" and tree["id"].startswith("e0:region:")
    states = [c for c in tree["children"] if c["type"] == "SDFGState"]
    assert states and states[0]["barrier"] is True  # a state is a barrier container
    assert len(states[0]["nests"]) == 2  # it holds both map-nests
    assert all("reads" in nest and "writes" in nest for nest in states[0]["nests"])


def test_cross_state_nests_are_blocked_and_name_the_region_merge():
    s = barred_session()
    a, b = (n["id"] for n in s.list_nests())
    reason = s.can_fuse(a, b)
    assert "different states" in reason and "merge the enclosing regions" in reason
    assert s.list_region_fusions(), "a state-fusion move must exist to unblock them"


def test_region_fusion_unblocks_cross_state_nest_fusion():
    s = barred_session()
    assert not s.list_fusions(), "nothing fuses while the state barrier stands"
    while s.list_region_fusions():  # merge regions to a fixed point (agent picks; here we drain)
        s.fuse_regions(s.list_region_fusions()[0]["id"])
    assert s.list_fusions(), "with the states merged, the two maps are now fusable"


def test_fuse_regions_bumps_epoch_and_stales_prior_ids():
    s = barred_session()
    moves = s.list_region_fusions()
    assert moves and moves[0]["kind"] == "fuse-states"
    epoch = s.epoch
    s.fuse_regions(moves[0]["id"])
    assert s.epoch == epoch + 1
    with pytest.raises(StaleHandle):  # the region-move id is now stale
        s.fuse_regions(moves[0]["id"])


# --- Phase 2/3: offload is a distinct axis from fusion --------------------------------------------


def test_offload_candidates_are_distinct_from_nest_fusion():
    s = make_session()
    cands = s.list_offload_candidates()
    assert cands and all(c["id"].startswith("e0:cand:") for c in cands)
    assert all("reads" in c and "writes" in c for c in cands)


def test_externalize_mints_nests_at_new_epoch_with_boundary_sets():
    s = make_session()
    nests = s.externalize()
    assert s.epoch == 1
    assert len(nests) == 2
    assert all(n["id"].startswith("e1:extnest:") for n in nests)
    producer = next(n for n in nests if n["writes"] == ["T"])
    assert producer["reads"] == ["A", "B"] and producer["symbols"] == ["N"]


def test_nest_boundary_exposes_abi_order_target():
    s = make_session()
    nest_id = s.externalize()[0]["id"]
    info = s.nest_boundary(nest_id)
    # boundary_order = inputs + outputs + symbols; set_kernel's abi_order is checked against it
    assert info["boundary_order"] == info["inputs"] + info["outputs"] + info["symbols"]


def test_set_kernel_sets_leaf_fields_without_bumping_epoch():
    s = make_session()
    nest_id = s.externalize()[0]["id"]
    epoch = s.epoch
    out = s.set_kernel(nest_id, "/abs/libk.a", "k", ["A", "B", "T", "N"])
    assert s.epoch == epoch  # a leaf-field write, ids stay valid
    assert out["abi_order"] == ["A", "B", "T", "N"]
    assert s.nest_boundary(nest_id)  # same id still resolves


def test_emit_reference_writes_numpy_oracle(tmp_path):
    s = Session(vertical_pair.to_sdfg(simplify=True), work_dir=str(tmp_path))
    nest_id = s.externalize()[0]["id"]
    path = s.emit_reference(nest_id)
    assert path.endswith(".py")
    with open(path) as f:
        assert "def " in f.read()
