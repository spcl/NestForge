# NormalizeMapBody._merge_siblings corrupts nested-state .sdfg backpointers -> InvalidSDFGNodeError (KeyError inside validate)

## Symptom

```
dace.sdfg.validation.InvalidSDFGNodeError: Node validation failed: '__rd1_dv' (at state BinOp_27, node loop_body)
Originating from source code at File "dace/transformation/interstate/loop_to_map.py", line 1032
```
caused by (chained via `from ex`):
```
KeyError: '__rd1_dv'
```
repr as raised: `InvalidSDFGNodeError("Node validation failed: '__rd1_dv'", SDFG (loop_body), 3, 7)`

Hit by the `hpcagent_bench` / npbench-polybench kernel `hpc/dense_linear_algebra/correlation/correlation` when canonicalized: `dace.transformation.passes.canonicalize.canonicalize(sdfg, target="cpu")`.

## Reproducer

```bash
OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp PMIX_MCA_gds=hash UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl DACE_default_build_folder=$HOME/.cache/dace_sweep/bugtriage PYTHONPATH=/home/primrose/Work/nest-forge python3 - <<'PY'
from nestforge.corpus import iter_dace_kernels, set_precision_fp64
from dace.transformation.passes.canonicalize import canonicalize

set_precision_fp64()
k = {kk.short_name: kk for kk in iter_dace_kernels()}["hpc/dense_linear_algebra/correlation/correlation"]
sdfg = k.to_sdfg(simplify=True)
sdfg.validate()               # passes
canonicalize(sdfg, target="cpu")   # InvalidSDFGNodeError(...'__rd1_dv'...)
PY
```

## Where it breaks

The exception is thrown from the canonicalize pipeline's internal `sdfg.validate()` call at `dace/transformation/passes/canonicalize/pipeline.py:1611` (inside `apply_pass`), which walks down through `dace/sdfg/validation.py:472` (`node.validate(sdfg, state, ...)`) into `dace/sdfg/nodes.py:811` (`NestedSDFG.validate`, computing `self.sdfg.free_symbols`) and `dace/sdfg/nodes.py:342` (`desc()`, `return sdfg.arrays[self.data]`) -- `sdfg.arrays` here does not contain `__rd1_dv`.

Confirmed live (catching the exception and inspecting the actual objects, not the serialized dump): the `NestedSDFG` codenode failing validation is the **middle-level** `loop_body` (built by `LoopToMap`'s first application, over `correlation`'s `__rd1_k1` loop) whose own `.arrays` genuinely lacks `__rd1_dv`. The states holding the `__rd1_dv` AccessNodes (`slice_data_30`, `BinOp_31`) belong to the **innermost** `loop_body` SDFG (built by `LoopToMap`'s second application, over the nested `__rd1_j0` loop) -- whose `.arrays` *does* contain `__rd1_dv` -- but `state.sdfg` for both of those states resolves to the wrong (middle-level) SDFG object:

```
node.sdfg (the codenode's own nested sdfg) name: loop_body  id: 125733974818160  arrays: [..., '__rd1_dv', ...]
state slice_data_30: st.sdfg is inner? False   id(st.sdfg) == id(failing_sdfg) (the MIDDLE loop_body, no '__rd1_dv')
state BinOp_31:      st.sdfg is inner? False   id(st.sdfg) == id(failing_sdfg)
```

Traced every assignment to `SDFGState.sdfg` (the `_sdfg` backpointer, `dace/sdfg/state.py:1358-1363`) for these two states across the whole canonicalize run. `LoopToMap.apply` (`dace/transformation/interstate/loop_to_map.py:1008,1013`, via `ControlFlowRegion.add_node` at `dace/sdfg/state.py:2950,2953`) sets them correctly to the innermost `loop_body` when it creates it. The **last** write before the failure is:

```
SET slice_data_30.sdfg = 'loop_body' (id=...)  <- pipeline.py:1607:apply_pass | normalize_map_body.py:185:apply_pass | normalize_map_body.py:252:_merge_siblings
SET BinOp_31.sdfg    = 'loop_body' (id=...)  <- pipeline.py:1607:apply_pass | normalize_map_body.py:185:apply_pass | normalize_map_body.py:252:_merge_siblings
```

`dace/transformation/passes/canonicalize/normalize_map_body.py`, `NormalizeMapBody._merge_siblings`, lines 250-252:

```python
sdutil.set_nested_sdfg_parent_references(base)
for blk in base.all_control_flow_blocks(recursive=True):
    blk.sdfg = base
```

`base.all_control_flow_blocks(recursive=True)` (`dace/sdfg/state.py:3070-3074`, via `all_control_flow_regions(recursive=True)` at `dace/sdfg/state.py:3029-3054`) descends **through `NestedSDFG` codenodes** (`state.py:3038-3043`: `if isinstance(node, nd.NestedSDFG): yield from node.sdfg.all_control_flow_regions(recursive=recursive, ...)`), so it yields every state at every nesting depth under `base` -- including states that belong to a *deeper* `NestedSDFG`'s own `.sdfg`, several levels down. The loop then stamps `blk.sdfg = base` on **all** of them unconditionally, overwriting the correct (deeper) backpointer with the shallow `base` reference. Contrast with the line immediately above it, `set_nested_sdfg_parent_references` (`dace/sdfg/utils.py:2367-2376`), which recurses the *same* structure but correctly sets each level's own direct parent (`node.sdfg.parent_sdfg = sdfg`, then recurses into `node.sdfg`) -- the adjacent, correct pattern the buggy loop should have followed instead of a flat `recursive=True` sweep.

## Root cause

Invariant violated: every `ControlFlowBlock`'s `.sdfg` backpointer must equal its *immediate* containing `SDFG` object (the one whose `.arrays`/`.symbols` actually scope its data references). `NormalizeMapBody._merge_siblings`'s cleanup loop (`normalize_map_body.py:251-252`) instead sets it to the outermost `base` of the merge for every control-flow block reachable transitively through nested `NestedSDFG` codenodes, corrupting the backpointer for any state that lives inside a deeper-nested SDFG underneath the merged siblings. Later code that resolves a state's owning SDFG via `state.sdfg` (e.g. `SDFGState.used_symbols`, `dace/sdfg/state.py:706,716`) then looks up array descriptors in the wrong (too-shallow) SDFG and KeyErrors on any name that is local to the true, deeper SDFG.

## Blast radius

Loud (`InvalidSDFGNodeError`, raised before codegen -- confirmed no silent wrong-answer path here since canonicalize's own internal validate() catches it first).

Trigger precondition: a map body deep enough to contain a multi-level `LoopToMap` nesting (an outer reduction loop whose body contains its own inner reduction loop, as in `correlation`'s two-level mean/stddev computation) *and* a `NormalizeMapBody` sibling-merge firing on a state at or above that nesting (`NormalizeMapBody` only merges when a map body has >=2 sibling `NestedSDFG`s, `normalize_map_body.py:183`). Full corpus/blast-radius count pending a scan of `hpcagent_bench`'s corpus under `canonicalize`; `correlation` is the one confirmed instance from the suspect list.

## Status

reproduced on dace commit c1ba4bf62, date 2026-07-27.
