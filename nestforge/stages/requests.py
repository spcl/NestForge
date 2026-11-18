# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transformation requests: an agent that finds no fitting move names one, with pairs of small programs before and
after it (``docs/design/transformation-requests.md``). A request changes nothing in the episode; it is checked here
and kept for the authoring step between rounds.

The checks are cheap and need no LLM: enough examples; every ``before`` and ``after`` is one ``@dace.program``; each
pair computes the same on random inputs; no pair is the same program after canonicalization (it would ask for
nothing); and no example copies the task's reference source."""

import copy
import inspect
import linecache
from collections.abc import Callable, Sequence
from typing import Any, Literal

import dace
import numpy as np
from dace.frontend.python.parser import DaceProgram

from nestforge.records import Record, Refusal, RefusalOutcome
from nestforge.stages.canonicalize import Targets, canonicalize

#: The fewest example pairs a request takes: fewer cannot pin down a pattern.
MIN_EXAMPLES = 5
#: The extent every symbol of an example takes when its pair is run on random inputs.
EXAMPLE_SIZE = 7
#: Consecutive lines of the task's reference source that, found in an example, refuse it as a copy of the task.
LEAK_LINES = 3
#: How close the outputs of a pair must be: the examples are small, so reassociation moves only the last bits.
RTOL, ATOL = 1e-9, 1e-12


class Example(Record):
    """One pair of a request: a small ``@dace.program`` before the transformation and the same program after it."""

    before: str
    after: str


class RequestResult(Record):
    """What :meth:`~nestforge.session.Session.request_transformation` did: ``recorded``, or ``refused`` with the reason
    and its class. Either way the program is unchanged."""

    status: Literal["recorded", "refused"]
    name: str
    rows: tuple[str, ...]
    reason: str = ""
    refusal: RefusalOutcome | None = None


class Request(Record):
    """A recorded request, as the authoring step reads it."""

    name: str
    description: str
    rows: tuple[str, ...]
    examples: list[Example]


class ExampleError(Exception):
    """An example that does not define, build or run one program; the message says which and why."""


def program_of(source: str, where: str) -> tuple[DaceProgram, dict[str, Any]]:
    """The one ``@dace.program`` ``source`` defines, and the namespace it was defined in.

    :raises ExampleError: The source does not run, or defines no program or several."""
    namespace: dict[str, Any] = {"dace": dace, "np": np, "numpy": np}
    # DaCe's frontend reads a program's source through inspect, which finds an exec'd module only in linecache
    filename = f"<{where}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    try:
        exec(compile(source, filename, "exec"), namespace)  # noqa: S102 -- the agent's example, run like write_kernel's code
    except Exception as err:  # noqa: BLE001 -- any failure of the example is its refusal, named
        raise ExampleError(f"{where} does not run: {type(err).__name__}: {err}") from err
    programs = [value for value in namespace.values() if isinstance(value, DaceProgram)]
    if len(programs) != 1:
        raise ExampleError(f"{where} defines {len(programs)} @dace.program functions, not one")
    return programs[0], namespace


def drawn(annotation: Any, sizes: dict[str, int], rng: np.random.Generator) -> Any:
    """A random value of one parameter's annotation: an array at ``sizes``, or a scalar."""
    if isinstance(annotation, dace.data.Array):
        shape = tuple(int(dace.symbolic.evaluate(extent, sizes)) for extent in annotation.shape)
        dtype = annotation.dtype.type
        if np.issubdtype(dtype, np.integer):
            return rng.integers(0, EXAMPLE_SIZE, size=shape).astype(dtype)
        return rng.standard_normal(shape).astype(dtype)
    if isinstance(annotation, dace.dtypes.typeclass):
        dtype = annotation.type
        return dtype(rng.integers(1, EXAMPLE_SIZE)) if np.issubdtype(dtype, np.integer) else dtype(rng.random() + 0.5)
    raise ExampleError(
        f"a parameter annotated {annotation!r}: annotate arrays as dace.float64[N] and scalars as dace.float64"
    )


def runnable(program: DaceProgram, namespace: dict[str, Any]) -> Callable[..., Any]:
    """The program's Python function, with every ``dace.symbol`` of its module bound to :data:`EXAMPLE_SIZE`."""
    for name, value in list(namespace.items()):
        if isinstance(value, dace.symbolic.symbol):
            namespace[name] = EXAMPLE_SIZE
    return program.f


def outputs_match(
    before: tuple[DaceProgram, dict[str, Any]], after: tuple[DaceProgram, dict[str, Any]], where: str
) -> str:
    """Empty when ``before`` and ``after`` leave the same values in every argument and return the same, on one draw of
    random inputs; else why not."""
    parameters = inspect.signature(before[0].f).parameters
    if list(parameters) != list(inspect.signature(after[0].f).parameters):
        return f"{where}: before and after take different parameters"
    symbols = {str(v): EXAMPLE_SIZE for v in before[1].values() if isinstance(v, dace.symbolic.symbol)}
    rng = np.random.default_rng(0)
    args = {name: drawn(p.annotation, symbols, rng) for name, p in parameters.items()}
    first, second = copy.deepcopy(args), copy.deepcopy(args)
    try:
        returned = (runnable(*before)(**first), runnable(*after)(**second))
    except Exception as err:  # noqa: BLE001 -- an example that fails on its inputs is refused, named
        return f"{where} fails on random inputs: {type(err).__name__}: {err}"
    for name in args:
        if not np.allclose(first[name], second[name], rtol=RTOL, atol=ATOL, equal_nan=True):
            return f"{where}: before and after leave different values in {name}"
    if not (returned[0] is None and returned[1] is None) and not np.allclose(
        np.asarray(returned[0]), np.asarray(returned[1]), rtol=RTOL, atol=ATOL, equal_nan=True
    ):
        return f"{where}: before and after return different values"
    return ""


def canonical_code(program: DaceProgram, targets: Targets) -> str:
    """The C++ DaCe generates for ``program`` after canonicalization, under one fixed name, so two programs that
    canonicalize alike compare equal."""
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, targets)
    sdfg.name = "example"
    return "\n".join(line.strip() for line in sdfg.generate_code()[0].clean_code.splitlines() if line.strip())


def copied_lines(text: str, reference: str) -> bool:
    """Whether ``text`` holds :data:`LEAK_LINES` consecutive non-blank lines of ``reference`` in order."""
    ours = [line.strip() for line in text.splitlines() if line.strip()]
    theirs = [line.strip() for line in reference.splitlines() if line.strip() and not line.strip().startswith("#")]
    windows = {tuple(theirs[i : i + LEAK_LINES]) for i in range(len(theirs) - LEAK_LINES + 1)}
    return any(tuple(ours[i : i + LEAK_LINES]) in windows for i in range(len(ours) - LEAK_LINES + 1))


def example_refusal(examples: Sequence[Example], targets: Targets, reference: str) -> str:
    """Empty when every example passes the checks of the module docstring; else the first failure."""
    if len(examples) < MIN_EXAMPLES:
        return f"a request takes at least {MIN_EXAMPLES} examples; it has {len(examples)}"
    for index, example in enumerate(examples):
        where = f"example {index}"
        if reference and (copied_lines(example.before, reference) or copied_lines(example.after, reference)):
            return f"{where} copies the task's reference source; write small programs of your own"
        try:
            before = program_of(example.before, f"{where} (before)")
            after = program_of(example.after, f"{where} (after)")
        except ExampleError as err:
            return str(err)
        # canonicalize first: running a pair binds its symbols to numbers
        try:
            same = canonical_code(before[0], targets) == canonical_code(after[0], targets)
        except Exception:  # noqa: BLE001 -- a pair DaCe cannot canonicalize is not refused for being the same
            same = False
        if same:
            return f"{where}: before and after are the same program after canonicalization, so it asks for nothing"
        mismatch = outputs_match(before, after, where)
        if mismatch:
            return mismatch
    return ""


def refused(name: str, rows: Sequence[str], reason: str, refusal: Refusal) -> RequestResult:
    return RequestResult(status="refused", name=name, rows=tuple(rows), reason=reason, refusal=refusal)
