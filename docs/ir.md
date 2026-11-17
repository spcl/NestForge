# The structure tree: NestForge's summary IR

[Overview](../README.md) · [Stages](stages.md) · [Code map](README.md)

NestForge owns one DaCe SDFG per session. Neither the agent nor a person reads the SDFG itself: they read the
**structure tree**, a text summary with one row per control-flow block, doall and library call. The tree is the IR
the agent reasons in. It names every row with a stable label, the label is how a move names what it rewrites, and the
tree prints again after every move. This page is the contract of that text: what a row is, which SDFG object it
stands for, what each field means, how bodies print in Python or C, and which calls change the program and which only
change the view.

Code: `nestforge/ir/introspect.py` renders the tree (`describe_graph`), `nestforge/ir/names.py` hands out the labels
(`Namer`), `nestforge/ir/emit_python.py` and `nestforge/ir/emit_c.py` print the bodies, `Session.describe` and
`Session.view` are the entry points, and `view_tree` is the agent's tool.

## An example

`axpy` repeated `T` times, then a reversed copy and a sum, after canonicalization. Collapsed, the program is one row:

```
SDFG 'axpy_then_reverse'
`- state2  doalls=3  reads=['x', 'y', 'z'] writes=['s', 'y', 'z']
```

`view_tree expand=[state2, state3, state4]` opens it; leaf bodies print as Python by default:

```
`- state2  reads=['x', 'y', 'z'] writes=['s', 'y', 'z']
   |- doall2  [_loop_it_0=0:T]  reduce=(+ over _loop_it_0 -> y)  reads=['x'] writes=['y']
   |  `- doall3  [_loop_it_1=0:N]  reduce=(+ over - -> y)  reads=['x'] writes=['y']  num_states=1
   |     : _wcr_priv__Add____out = 2.0 * x[_loop_it_1]
   |     : y[_loop_it_1] = y[_loop_it_1] + _wcr_priv__Add____out
   |     `- state3  reads=['x'] writes=['_wcr_priv__Add____out']
   |- doall4  [_loop_it_2=0:N]  reads=['y'] writes=['z']  num_states=1
   |  : z[_loop_it_2] = y[N - _loop_it_2 - 1] * 3.0
   |  `- state4  reads=['y'] writes=['z']
   `- lib_reduce0  LIBNODE Reduce [CPU]  reads=['z'] writes=['s_slice']
```

`view_tree c=[state2]` prints the same bodies as C:

```
   |     : _wcr_priv__Add____out[0] = (2.0 * x[_loop_it_1]);
   |     : y[_loop_it_1] += _wcr_priv__Add____out[0];
   ...
   |  : z[_loop_it_2] = (y[((N - _loop_it_2) - 1)] * 3.0);
```

## Rows

Every row is one SDFG object. Children print below their parent in execution order (`names.in_order`), drawn with
`|-` and `` `- ``. The order is topological, ties broken by insertion order, so it is the same on every run.

| Row | SDFG object | Prints |
|---|---|---|
| `SDFG '<name>'` | the root `SDFG` | the header |
| `for<n>  <var>=<lo>:<hi>` | `LoopRegion` | a sequential loop: its iterations run in order (`for` and `while` alike); an innermost loop adds `num_states=` |
| `if<n>`, then `<branch>  when <cond>` / `else` | `ConditionalBlock` and its branch regions | the first branch whose condition holds runs, so stored order is execution order |
| `state<n>` | `SDFGState` | one dataflow step; collapsed it adds `doalls=<k>` (top-level maps and library calls) |
| `doall<n>  [<p>=<lo>:<hi>[:<step>], ...]` | `MapEntry` | a map: iterations proven independent, run in any order or at once |
| `lib_<family><n>  LIBNODE <Kind> [<impl>]` | `LibraryNode`, or a one-element copy's tasklet | a library call, a black-box doall; `[<impl>]` is its implementation (`auto` until chosen), none on an `ExternalCall`. DaCe makes a one-element copy outside every map the tasklet `_cpy_out = _cpy_in`, what a `Copy` node's automatic expansion becomes, so it prints `lib_copy<n>  LIBNODE Copy [tasklet]` and the `lib-*` moves take it like any copy |
| `: <statement>` | the body of a leaf doall | what one iteration computes, in the row's body language |

A doall whose body is a nested SDFG (canonical form puts most bodies there) shows that SDFG's blocks as rows below
it once open, so `subgraph-fission` can name the state it cuts after. A nest inside an inner doall prints under that
doall only.

### Fields

| Field | On | Means |
|---|---|---|
| `reads=[..] writes=[..]` | state, doall, library call | the containers it reads and writes, sorted. A state leaves out the transients no other state touches (`introspect.outlives`): a temporary of one state is not part of its interface |
| (none) | state, loop, doall | work and depth are not printed: `get_work_depth labels=[...]` gives them per row (DaCe's work-depth analysis, one pass per program, at most 60 s), at the session's sizes or as formulas (`formula=true`), simplified under the program's facts. A library call's own cost is a free symbol (`lib_tileops2_work`) |
| `doalls=<k>` | collapsed state | how many top-level doalls and library calls it holds |
| `num_states=<k>` | innermost loop, doall holding a nest | the states under it, nested SDFGs included |
| `reduce=(<op> over <axes> -> <target>)` | doall | an accumulation leaving the doall (a WCR edge on its exit): `+`, `*` or the reduction's name, the parameters the written subset does not mention (`-`: none, an element-wise accumulation), the array it lands in |
| `[<impl>]` | library call | its implementation: the fastest one this host has for the target, never `pure` while another exists, until `set-library-implementation` pins one |

Symbolic text is DaCe's: `int_floor(a, b)` is floor division, `Max`/`Min` as in SymPy, and canonical form names map
parameters `_loop_it_<k>`.

## Labels

A label is `<kind><n>` (`for`, `if`, `state`, `doall`, `lib_<family>`): unique across the whole program, nested SDFGs
included, and stored on the object itself (a state's or region's label, a map's label, a library node's label), so it
survives a deep copy. `Namer` hands them out once: a label is never reused, even after its row is gone.

A row keeps its label while it lives. A move retires the labels of the rows it consumed and names whatever it made
afresh (`Session.bump`), so a label read before a move either still names the same row or is refused as unknown.
Labels are names, not positions: nothing about them is computed from the tree's shape.

## Bodies: Python or C

A leaf doall's body prints under it as `: ` lines, in the body language of its row.

- **Python** (the default, `emit_python.body_lines`). Printed from a lowered copy of the program (`lowered_twin`),
  NumPy-flavoured: one assignment per statement, accumulations as `y[i] = y[i] + v`, library calls by their NumPy
  equivalent where one exists. A construct Python cannot spell prints as `<not emitted: <why>>`, the main one being a
  C++ tasklet (a vectorized body's tile calls, inline assembly).
- **C** (`emit_c.body_c`). Printed from the live program, nothing lowered: a Python tasklet is spelled by DaCe's C++
  unparser, a C++ tasklet as written, so vector intrinsics and inline assembly show as they are. Each connector is
  replaced by the element its memlet names (`x[_loop_it_1]`), or the address of the first element when the memlet
  spans more than one (`&x[_loop_it_0]`, a tile's lanes). An accumulation is a compound assignment (`+=`, `*=`, `&=`,
  `|=`, `^=`), `min`/`max`, or a call of its lambda. A nested map becomes C `for` loops, a nested SDFG its control
  flow (`for`, `if`/`else`, interstate assignments as statements, under the edge's condition when it has one), a
  library call a comment naming its kind and implementation.

The language is set per row and inherited: a row prints in its own setting, else that of the nearest row around it,
else Python. So `c=[state2]` prints every body in `state2` as C, and `python=[doall4]` then takes `doall4` back to
Python alone.

## View calls change nothing

Two kinds of call act on the tree. View calls choose what is shown; program calls (moves, defaults, scopes, kernel
writes) rewrite the SDFG. A view call never touches the SDFG: the tests check that the program's hash is the same
before and after printing.

| Call (`view_tree` argument) | Session | Does |
|---|---|---|
| `expand=[labels]` | `Session.expand` | opens states and doalls: an open state shows its doalls as rows; an open doall shows its statements, nested doalls and the states of its body. A doall of an open state stays closed until opened itself |
| `collapse=[labels]` | `Session.collapse` | closes open states and doalls, one opened by a row around it included; the row stays, its content goes |
| `c=[labels]` | `Session.set_language(labels, "c")` | prints the bodies under the rows as C |
| `python=[labels]` | `Session.set_language(labels, "python")` | prints them as Python again |

`view_tree` applies them in that order and prints the tree after them, each call's note first (`ok: ...`,
`illegal: ...`, `not open: ...`). The view state is two maps, `Session.expanded` (label -> open) and
`Session.languages` (label -> language). An entry lasts while its row lives: a move keeps the entries of the rows it
does not retire and drops the others. Rolling a move back restores the view the program had.

## What the tree leaves out

The tree is a summary, so some of the SDFG is not on it: memlet subsets other than in bodies, access nodes, the
ordering edges between nodes, interstate conditions other than a loop's domain and a branch's condition, and the
storage and schedule of each container. A move reads all of it: legality is DaCe's check on the full SDFG, never the
tree's. For the whole picture of one kernel, its CPF source (`read_kernel`) is the program text DaCe compiles.
