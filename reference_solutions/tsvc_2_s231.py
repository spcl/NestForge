# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference solution of TSVC s231: ``python -m nestforge.replay reference_solutions/tsvc_2_s231.py``.

``aa[j, i] = aa[j-1, i] + bb[j, i]``: the recurrence runs along ``j``, ``i`` is independent. After stage 1 the
program is ``for j { map i }``, one parallel region per ``j``. The one offered move, ``interchange-loop-map``, gives
``map i { for j }``: one parallel region, each thread walking a column. It wins with many threads (fewer regions) and
loses on one thread (the column walk is strided by ``LEN_2D``). On the GPU ``default_moves()`` already takes it."""

from nestforge.replay import Context, Script

#: The corpus kernel (``nestforge.corpus.bench.corpus_kernel`` name).
KERNEL = "tsvc_2_s231"

#: Where the gap is measured: a threaded CPU, a serial CPU (where the move is the trap), the GPU.
CONFIGS = (Context("cpu", 72, "M"), Context("cpu", 1, "M"), Context("gpu", 1, "M"))

#: The j loop (writes aa, reads bb) into the i map.
INTERCHANGE = "interchange-loop-map"


def interchange(s: Script, ctx: Context) -> None:
    s.apply(s.pick(INTERCHANGE, reads=["bb"], writes=["aa"]))
    s.finish()


def no_move(s: Script, ctx: Context) -> None:
    s.finish()


def solve(s: Script, ctx: Context) -> None:
    """Interchange unless the CPU runs one thread."""
    if ctx.device == "gpu" or ctx.threads > 1:
        interchange(s, ctx)
    else:
        no_move(s, ctx)


#: The schedule the reference rejects on each configuration, timed next to it.
DECOYS = {"interchange": interchange, "no-move": no_move}
