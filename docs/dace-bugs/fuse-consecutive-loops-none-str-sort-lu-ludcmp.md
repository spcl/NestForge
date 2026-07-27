# FuseConsecutiveLoops._state_signature: sorted() on mixed None/str connector fields -> TypeError

## Symptom

```
TypeError: '<' not supported between instances of 'NoneType' and 'str'
```
raised from:
```
File "dace/transformation/passes/canonicalize/fuse_consecutive_loops.py", line 250, in _state_signature
    return (tuple(node_sig), tuple(sorted(edge_sig)))
```

Hit by the `hpcagent_bench` / npbench-polybench kernels `hpc/dense_linear_algebra/lu/lu` and `hpc/dense_linear_algebra/ludcmp/ludcmp` when canonicalized: `dace.transformation.passes.canonicalize.canonicalize(sdfg, target="cpu")`. Both kernels fail identically (same failing state label `BinOp_15`, same 12-tuple `edge_sig`, byte-for-byte) since `ludcmp` embeds the same LU-factorization loop body as `lu`.

## Reproducer

```bash
OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp PMIX_MCA_gds=hash UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl DACE_default_build_folder=$HOME/.cache/dace_sweep/bugtriage PYTHONPATH=/home/primrose/Work/nest-forge python3 - <<'PY'
from nestforge.corpus import iter_dace_kernels, set_precision_fp64
from dace.transformation.passes.canonicalize import canonicalize

set_precision_fp64()
kernels = {k.short_name: k for k in iter_dace_kernels()}
for short in ("hpc/dense_linear_algebra/lu/lu", "hpc/dense_linear_algebra/ludcmp/ludcmp"):
    sdfg = kernels[short].to_sdfg(simplify=True)
    sdfg.validate()               # passes
    canonicalize(sdfg, target="cpu")   # TypeError: '<' not supported between instances of 'NoneType' and 'str'
PY
```

## Where it breaks

`dace/transformation/passes/canonicalize/fuse_consecutive_loops.py`, `FuseConsecutiveLoops._state_signature` (called from `_bodies_match:210-211`, called from `_adjacent_identical:189`, called from `_fuse_one:166`, called from `apply_pass:135`, run by the canonicalize pipeline).

`_state_signature` (lines 238-250) builds one tuple per state edge:

```python
edge_sig.append((_node_key(e.src, loop_var, local_scratch), e.src_conn,
                 _node_key(e.dst, loop_var, local_scratch), e.dst_conn, data_name, subset, wcr))
...
return (tuple(node_sig), tuple(sorted(edge_sig)))
```

Traced live (instrumented `_state_signature` to dump `edge_sig` for `lu`'s failing state `BinOp_15` before the `sorted()` call). Two of the 12 tuples are:

```
(('access', 'A'), None, ('access', '__scratch__'), 'views', 'A', '_loop_it_0, 0:__lv__', 'None')
(('access', 'A'), None, ('access', '__scratch__'), None,    'A', '__lv__, __lv__',        'None')
```

Both edges go `AccessNode 'A' -> (local-scratch) AccessNode`, so index 0 (`_node_key(e.src)` = `('access','A')`), index 1 (`e.src_conn` = `None`), and index 2 (`_node_key(e.dst)` = `('access','__scratch__')`, since `_canon_data` at line 92-95 maps every body-local scratch array to the same placeholder regardless of which underlying array it is) are all equal between the two tuples. `sorted()` must then compare index 3, `e.dst_conn`: one edge targets a DaCe View AccessNode's reserved `'views'` connector (a `str`), the other is a plain AccessNode-to-AccessNode copy edge whose connector is `None`. Python 3's tuple comparison has no total order between `NoneType` and `str`, so `sorted(edge_sig)` raises.

## Root cause

Invariant violated: `_node_key` (`fuse_consecutive_loops.py:98-104`) collapses every `AccessNode` to `('access', canonical_data_name)`, deliberately erasing which physical array/connector convention it uses (that is the point of the scratch-name canonicalization, per the module's own docstring on `_SCRATCH_PLACEHOLDER`). But it does not account for DaCe's View convention, where an edge into a `dace.data.View` AccessNode carries the fixed connector name `'views'` (`str`) while an ordinary AccessNode-to-AccessNode copy edge carries `None`. Two edges that tie on `_node_key(src)`/`src_conn`/`_node_key(dst)` can therefore differ in *type* (not just value) at the `dst_conn` position, and `_state_signature` sorts the raw `edge_sig` tuples (line 250) with no type-normalizing sort key, so any such tie crashes `sorted()` outright on `None < str`.

## Blast radius

Loud (`TypeError`, raised before codegen; canonicalize aborts with a hard exception, no wrong-answer path).

Confirmed identical failure on both `lu` and `ludcmp` (2/2 kernels checked from the suspect list). Trigger precondition: a loop body containing >= 2 AccessNodes that canonicalize to the same generic `_node_key` (e.g. two body-local scratch nodes, or two nodes of the same generic `('other', type(node).__name__)` kind such as two distinct `MapEntry`/`MapExit` pairs) where one of the tied pair's boundary edges is a View connector (`'views'`) and another is a plain (`None`) connector -- i.e. any kernel whose canonicalizable loop body slices an array into a view (`A[i, 0:n]`-style row/column view, common in dense linear-algebra kernels) alongside a plain local scratch copy. Not scanned beyond `lu`/`ludcmp`; likely to recur on other polybench dense-linear-algebra kernels with the same tiled/view-slicing shape (`gaussian`, `cholesky`, `gramschmidt` use similar row/column view patterns) but this was not verified.

## Status

reproduced on dace commit c1ba4bf62, date 2026-07-27.
