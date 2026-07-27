# MapFission (expr_index=0) orphans a map-exit AccessNode -> InvalidSDFGNodeError('Isolated node')

## Symptom

```
dace.sdfg.validation.InvalidSDFGNodeError: Isolated node
```
repr as raised: `InvalidSDFGNodeError('Isolated node', SDFG (nestforge_tsvc2_corpus_s221_d_single), 0, 3)`

Hit by nest-forge's TSVC2 kernel `s221` (and 3 others, see Blast radius) at `nestforge.granularity.to_canonical_atoms` -- called by every rung of `fuse_first_k(k)` (atoms through maximal), so `s221` fails the fusion-granularity sweep at every rung, on any backend, because they all start from the same broken P0 canonical-atoms base.

## Reproducer

```bash
OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp PMIX_MCA_gds=hash UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl DACE_default_build_folder=$HOME/.cache/dace_sweep/bugtriage PYTHONPATH=/home/primrose/Work/nest-forge python3 - <<'PY'
from nestforge import tsvc
from nestforge.granularity import to_canonical_atoms

k = [x for x in tsvc.iter_tsvc_kernels() if x.key == 's221'][0]
sdfg = tsvc.build_sdfg(k, 'canonicalize')
sdfg.validate()                 # passes
to_canonical_atoms(sdfg)        # == fission_to_statements(sdfg)
sdfg.validate()                 # InvalidSDFGNodeError('Isolated node', ..., 0, 3)
PY
```

Bisection: `build_sdfg(k, 'canonicalize')` validates. `nestforge.fission_arms.fission_to_statements` runs three steps in order (`SplitStatements(split_maps=True)`, `LoopFission`, `fission_multi_output_maps`); applying them one at a time shows `SplitStatements` and `LoopFission` each apply 0 times and the SDFG still validates after both. `fission_multi_output_maps` applies exactly once and the result fails to validate. That one application is `MapFission.apply_to(sdfg, expr_index=0, map_entry=<single_state_body_0_map>)` -- the "map with arbitrary subgraph" pattern (`nestforge/fission_arms.py:78-79`), not the nested-SDFG pattern.

The isolated node is the SECOND `AccessNode` named `a` in state `single_state_body` (node index 3): before fission it has 1 in-edge (from the old `MapExit`'s `OUT_2` connector, memlet `a[1:LEN_1D]`) and 0 out-edges -- this is `s221`'s **write-back** access node for array `a`. After fission it has 0 in-edges and 0 out-edges.

## Where it breaks

Raised at `dace/sdfg/validation.py:487` (`raise InvalidSDFGNodeError("Isolated node", sdfg, state_id, nid)`, guarded by the isolated-node check at `validation.py:482`).

Produced by `MapFission.apply()`, `dace/transformation/dataflow/map_fission.py`, `expr_index == 0` path:

- `external_edges_exit = list(state.in_edges(map_exit))` (`map_fission.py:370`) and the `edge_to_outer` map built from it (`map_fission.py:407-415`) correctly record that the edge `__map_fusion_a -> map_exit(IN_2)` should be replaced by `map_exit(OUT_2) -> a` (the write-back AccessNode).
- The only two places that ever *consume* `edge_to_outer` to create a replacement edge are: (1) the per-component `component_out` loop (`map_fission.py:494-522`), which only looks at `state.out_edges(component_out)` for `component_out` in `_components(subgraph)` -- i.e. edges whose **source** is a `CodeNode`/`EntryNode`, and (2) the "other sources/sinks" loop (`map_fission.py:525-543`), which only rewires nodes in `subgraph.source_nodes()` / `subgraph.sink_nodes()`.
- `__map_fusion_a` is an `AccessNode` that is neither: it has an internal out-edge (feeds `_assign_in_a_to_b_slice_plus_a_slice` inside the subgraph, so it is not a *sink*) **and** an external out-edge straight to the old `map_exit` (so its boundary edge is never attached to any `component_out`). Confirmed directly: `mfa in subgraph.sink_nodes()` is `False`, `mfa in subgraph.source_nodes()` is `False`, and of its 2 out-edges, `__map_fusion_a -> map_exit(IN_2)` is not a member of `subgraph.edges()` (i.e. it is genuinely a border edge) yet is touched by neither reconnection loop.
- `graph.remove_nodes_from([map_entry, map_exit])` (`map_fission.py:684`) then deletes `map_exit`, which silently deletes that unreplaced edge as a side effect (networkx semantics), leaving the write-back `a` AccessNode with degree 0.

## Root cause

Invariant violated: MapFission's `expr_index=0` reconnection logic assumes every boundary crossing of the fissioned subgraph is realized either (a) at a `component_in`/`component_out` node directly, or (b) at a bona-fide subgraph source/sink `AccessNode`. It does not handle a **border `AccessNode` that has both an internal successor/predecessor edge and a direct external edge to the (about-to-be-removed) `map_entry`/`map_exit`** -- exactly the shape produced when the map's output side goes `component_out -> border_AccessNode -> map_exit` (the `__map_fusion_a` pattern nest-forge's own maximal-fusion / WAR-hazard renaming creates for `s221`'s `a`, which is both read and written by the map). That edge is computed into `edge_to_outer` but never consumed, then destroyed as collateral damage of `map_exit`'s removal.

## Blast radius

Loud (validate() raises `InvalidSDFGNodeError`; no silent wrong-answer path found -- the graph never reaches codegen).

Scanned nest-forge's `tsvc2` corpus (151 kernels: `build_sdfg(k, 'canonicalize')` then `to_canonical_atoms` then `validate()`, one kernel at a time, no compilation):
- 137 OK
- 4 hit this exact `InvalidSDFGNodeError('Isolated node', ...)` at the same `fission_multi_output_maps` call: `s152`, `s221`, `s241`, `s243`
- 10 hit a different, pre-existing failure (`'Data descriptor X is written to, but only given to nested SDFG as an input connector'`, and 2x `StopIteration`) -- not this bug, not investigated further here.

Not scanned: `tsvc2_5`, `foundation`, or the polybench/npbench (`hpcagent_bench`) corpus for this specific bug, so the true count across all of nest-forge's corpora is >= 4.

Because `nestforge.granularity.to_canonical_atoms` is the P0 base every `GranularityPoint` in the fusion ladder normalizes from (`nestforge/granularity.py:42-47,75-77`), all 4 affected kernels fail at **every** granularity rung (`atoms` through `maximal`), matching the reported symptom.

## Status

reproduced on dace commit c1ba4bf62, date 2026-07-27.
