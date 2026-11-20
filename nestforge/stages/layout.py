# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``relayout``: lay a program argument out as one of relayout's request operations, in the bench's own
``requests`` JSON. The program is rewritten to work on the argument as laid out (relayout's ``apply_requests``,
converting nothing), and the operations go to the bench as the submission's ``requests``: the harness lays every
argument out before the call and reads the outputs back after it, as NestForge does when it times the program."""

import copy
from collections.abc import Mapping, Sequence

import dace
import numpy as np
from dace import data
from relayout.convert import from_layout, to_layout
from relayout.expr import eval_index
from relayout.layout import IndexMap
from relayout.ops import Action, ArraySpec, Cast, Operation, Requests
from relayout.passes.sdfg.apply_requests import apply_requests
from relayout.passes.single_node import unapplied
from relayout.requests import parse_operation, requests_json

__all__ = [
    "LAYOUT_ACTIONS",
    "LOCAL_ACTIONS",
    "RELAYOUT",
    "argument_layouts",
    "laid_out",
    "layout_operation",
    "layout_refusal",
    "listed_operation",
    "local_operation",
    "locally_laid_out",
    "names_locals",
    "logical_outputs",
    "physical_inputs",
    "requests_of",
]

#: The move kind, on no row, that lays an argument out; it takes ``operation``.
RELAYOUT = "relayout"

#: The operations the move takes: every single-node operation relayout rewrites a program for, except ``zip`` and
#: ``unzip``, which change the argument list (``format``, ``distribute``, ``replicate`` and ``place`` are the caller's).
LAYOUT_ACTIONS: tuple[Action, ...] = (
    Action.CAST,
    Action.BLOCK,
    Action.UNBLOCK,
    Action.SPLIT,
    Action.UNSPLIT,
    Action.PERMUTE,
    Action.PAD,
    Action.UNPAD,
    Action.SHUFFLE,
    Action.LINEAR_SHUFFLE,
)


#: The operations the move takes on an array the program makes itself: every digit operation, and ``zip`` / ``unzip``
#: (array of structs, struct of arrays), which change no argument list there. A local's type changes with
#: ``change-precision``, so ``cast`` stays the arguments'.
LOCAL_ACTIONS: tuple[Action, ...] = (
    *(action for action in LAYOUT_ACTIONS if action is not Action.CAST),
    Action.ZIP,
    Action.UNZIP,
)


def locals_of(program: dace.SDFG) -> dict[str, data.Array]:
    """The arrays the program makes itself (transients, not views), by name."""
    return {
        name: desc
        for name, desc in program.arrays.items()
        if isinstance(desc, data.Array) and not isinstance(desc, data.View) and desc.transient
    }


def names_locals(raw: Mapping[str, object], program: dace.SDFG) -> bool:
    """Whether the operation ``raw`` names only arrays ``program`` makes itself (else it lays out arguments)."""
    try:
        op = parse_operation(dict(raw), "operation")
    except ValueError:
        return False
    existing = Requests(operations=(op,)).named() & set(program.arrays)
    return bool(existing) and existing <= set(locals_of(program))


def local_operation(raw: Mapping[str, object], program: dace.SDFG) -> Operation:
    """``raw`` as an operation on arrays ``program`` makes itself.

    :raises ValueError: ``raw`` is no such operation, with the reason."""
    op = parse_operation(dict(raw), "operation")
    if op.action not in LOCAL_ACTIONS:
        raise ValueError(
            f"relayout takes one of {[a.value for a in LOCAL_ACTIONS]} on an internal array; not {op.action.value!r} "
            "(an internal array's type changes with change-precision)."
        )
    return op


def locally_laid_out(program: dace.SDFG, op: Operation) -> dace.SDFG:
    """A copy of ``program`` with its own array(s) ``op`` names laid out in place: nothing is converted, since no
    caller sees them.

    :raises ValueError: relayout refuses ``op`` on ``program``, with the reason."""
    requests = Requests(operations=(op,))
    refusal = unapplied(requests)
    if refusal is not None:
        raise ValueError(refusal)
    laid = copy.deepcopy(program)
    apply_requests(laid, requests, convert=False)
    return laid


def arguments(program: dace.SDFG) -> dict[str, data.Array]:
    """The program's array arguments, by name."""
    return {
        name: desc
        for name, desc in program.arrays.items()
        if isinstance(desc, data.Array) and not isinstance(desc, data.View) and not desc.transient
    }


def layout_operation(raw: Mapping[str, object], program: dace.SDFG) -> Operation:
    """``raw`` as an operation on an argument of ``program`` (the program before its first layout).

    :raises ValueError: ``raw`` is no such operation, with the reason."""
    op = parse_operation(dict(raw), "operation")
    if op.action not in LAYOUT_ACTIONS:
        raise ValueError(f"relayout takes one of {[a.value for a in LAYOUT_ACTIONS]}; not {op.action.value!r}.")
    names = set(arguments(program))
    named = Requests(operations=(op,)).named()
    if not named <= names:
        raise ValueError(
            f"relayout lays out the program's array arguments {sorted(names)}; not {sorted(named - names)}."
        )
    return op


def listed_operation(program: dace.SDFG) -> dict[str, object] | None:
    """The operation ``relayout`` is listed with, on ``program`` before its first layout: the first array argument of
    two or more dimensions with its two innermost digits swapped (a transpose, what a strided read wants), or None
    when no argument has two. Any other operation is the agent's to pass."""
    for name, desc in arguments(program).items():
        rank = len(desc.shape)
        if rank >= 2:
            order = [f"i{digit}" for digit in range(rank)]
            order[-2], order[-1] = order[-1], order[-2]
            return {"action": Action.PERMUTE.value, "array": name, "order": order}
    return None


def laid_out(program: dace.SDFG, operations: Sequence[Operation]) -> dace.SDFG:
    """A copy of ``program`` working on its arguments as ``operations`` lay them out.

    :raises ValueError: relayout refuses ``operations`` on ``program``, with the reason."""
    requests = Requests(operations=tuple(operations))
    refusal = unapplied(requests)
    if refusal is not None:
        raise ValueError(refusal)
    laid = copy.deepcopy(program)
    apply_requests(laid, requests, convert=False)  # a LayoutError is a ValueError naming the use it cannot carry
    return laid


def argument_layouts(program: dace.SDFG, operations: Sequence[Operation]) -> dict[str, tuple[IndexMap, np.dtype]]:
    """Each argument ``operations`` touch, of ``program`` before its first layout: its layout and element type."""
    args = arguments(program)
    specs = [ArraySpec(n, len(d.shape), element=np.dtype(d.dtype.as_numpy_dtype()).name) for n, d in args.items()]
    requests = Requests(operations=tuple(operations))
    state = requests.run(specs)
    return {
        name: (state.layout(name, [str(e) for e in args[name].shape]), np.dtype(state.arrays[name].element_type))
        for name in sorted(requests.named())
    }


def physical_inputs(
    inputs: Mapping[str, np.ndarray], layouts: Mapping[str, tuple[IndexMap, np.dtype]], sizes: Mapping[str, int]
) -> dict[str, np.ndarray]:
    """``inputs`` as the harness passes them: every laid-out argument laid out and cast."""
    return {
        name: to_layout(np.asarray(value), layouts[name][0], sizes, layouts[name][1]) if name in layouts else value
        for name, value in inputs.items()
    }


def logical_outputs(
    buffers: Mapping[str, np.ndarray], layouts: Mapping[str, tuple[IndexMap, np.dtype]], sizes: Mapping[str, int]
) -> dict[str, np.ndarray]:
    """The flat ``buffers`` a run left, every laid-out argument read back into its logical shape."""
    out = dict(buffers)
    for name, (layout, _) in layouts.items():
        if name in out:
            shape = tuple(eval_index(extent, sizes, f"array {name!r}") for extent in layout.extents())
            out[name] = from_layout(out[name].reshape(shape), layout, sizes)
    return out


def layout_refusal(operations: Sequence[Operation]) -> str | None:
    """Why the program cannot be assembled for the bench with ``operations``, or None."""
    cast = sorted({op.array for op in operations if isinstance(op, Cast)})
    if cast:
        return (
            f"a cast ({cast}) changes an argument's element type, which the bench's entry for this program still "
            "declares at the kernel's precision; roll the cast back to finalize."
        )
    return None


def requests_of(operations: Sequence[Operation]) -> dict[str, object]:
    """The submission's ``requests`` for ``operations`` (``{}``: none)."""
    requests: dict[str, object] = requests_json(Requests(operations=tuple(operations)))
    return requests
