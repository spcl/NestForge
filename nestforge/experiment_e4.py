"""E4 driver (paper C4): quality vs cost, traditional search against the scoped agent.

C4 claims both optimizer classes drive the SAME granularity surface, and that the agent reaches
near-oracle quality with markedly fewer measurements. That is a two-number claim per run -- what it found
(quality) and what it spent to find it (cost) -- so E4 records both in one ledger and never reports one
without the other. A search that is cheap because it stopped early is not a win, and a search that is good
because it measured everything is not one either.

The oracle is exhaustive search: it measures every rung, so its winner IS the best rung by construction.
Every other strategy is scored against it -- ``quality = oracle_us / best_us``, 1.0 meaning the strategy
found the optimum, below 1.0 meaning it settled for something slower. Cost is
:class:`~nestforge.sweep.MeasureLedger` measurements, the count of whole-program builds the strategy asked
for, which is the quantity that actually costs hours on a corpus.

Measurements are cached across strategies within a kernel/backend, because a rung's time does not depend
on who asked for it. The ledger still counts every REQUEST, so the cost numbers stay the honest
"measurements this strategy needed" -- caching makes E4 affordable to run, it does not make a strategy
look cheaper than it is.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from nestforge import tsvc
from nestforge.arena import discover_compilers
from nestforge.experiment_e1 import run_e1_cell
from nestforge.granularity import granularity_ladder
from nestforge.policy import exhaustive_search, hillclimb_search
from nestforge.sweep import MeasureLedger

#: The two optimizer classes C4 compares, over one shared surface: the traditional oracle and the scoped
#: agent policy. Both take ``(labels, measure, ledger)`` and return a ``SearchResult``.
STRATEGIES: Dict[str, Callable] = {"exhaustive": exhaustive_search, "hillclimb": hillclimb_search}

#: The strategy whose winner defines the optimum. It measures every rung, so it cannot be beaten on
#: quality -- which is exactly what makes it the reference and never the "winner" of E4.
ORACLE = "exhaustive"


@dataclass(frozen=True)
class E4Row:
    """One (kernel, backend, strategy) run. ``quality`` is ``oracle_us / best_us`` (1.0 = found the
    optimum); ``measurements`` is what it spent. ``ok`` is False when the strategy never measured a valid
    rung, and ``error`` says why."""
    kernel: str
    backend: str
    strategy: str
    best: str
    best_us: float
    quality: float
    measurements: int
    tokens: int
    ok: bool
    error: Optional[str] = None


def run_e4(kernels: Sequence[tsvc.TsvcKernel],
           out_dir,
           unit: str = "map",
           max_granularity_points: int = 4,
           opt_mode: str = "canonicalize",
           backends: Optional[Dict[str, str]] = None,
           strategies: Optional[Dict[str, Callable]] = None,
           reps: int = 7,
           preset: str = "S",
           seed: int = 0) -> List[E4Row]:
    """Run every strategy over the same granularity ladder, per (kernel, backend), and ledger both sides.

    The ladder, the measurement, and its knobs are identical across strategies -- the only thing that
    varies is which rungs the strategy chooses to measure, which is the comparison C4 wants. A rung that
    fails to build measures as ``inf``, so a strategy can still finish and simply not win there.
    """
    out_dir = Path(out_dir)
    backends = backends or discover_compilers()
    strategies = strategies or STRATEGIES
    rows: List[E4Row] = []
    caught = Exception  # recorded per row, for the sweep's reason (see run_e1)
    for kernel in kernels:
        try:
            canonical = tsvc.build_sdfg(kernel, opt_mode)
            ladder = granularity_ladder(canonical, max_points=max_granularity_points)
        except caught as e:
            for backend_name in backends:
                for name in strategies:
                    rows.append(E4Row(kernel.key, backend_name, name, "-", float("inf"), 0.0, 0, 0, False, repr(e)))
            continue
        points = {p.name: p for p in ladder}
        labels = list(points)
        for backend_name, backend_path in backends.items():
            cache: Dict[str, float] = {}

            def measure(label: str, _bn=backend_name, _bp=backend_path, _k=kernel, _c=canonical) -> float:
                # Cached per (kernel, backend): a rung's time does not depend on which strategy asked, and
                # the ledger counts requests, not cache misses -- so sharing cannot flatter anyone.
                if label not in cache:
                    cell = run_e1_cell(_k,
                                       _bn,
                                       _bp,
                                       points[label],
                                       out_dir / _k.key / _bn / label,
                                       unit,
                                       opt_mode,
                                       reps,
                                       canonical=copy.deepcopy(_c),
                                       preset=preset,
                                       seed=seed)
                    cache[label] = cell.median_us if cell.ok else float("inf")
                return cache[label]

            results = {}
            for name, search in strategies.items():
                try:
                    results[name] = search(labels, measure, MeasureLedger())
                except caught as e:
                    rows.append(E4Row(kernel.key, backend_name, name, "-", float("inf"), 0.0, 0, 0, False, repr(e)))
            oracle_us = results[ORACLE].best_us if ORACLE in results else float("inf")
            for name, res in results.items():
                rows.append(score(kernel.key, backend_name, name, res, oracle_us))
    return rows


def score(kernel: str, backend: str, strategy: str, result, oracle_us: float) -> E4Row:
    """One row from a finished search. Quality needs BOTH the strategy's time and the oracle's to be finite
    -- when every rung failed to build there is no optimum to be near, and reporting 1.0 there would claim
    the strategy matched an oracle that itself found nothing."""
    best_us = result.best_us
    usable = 0.0 < best_us < float("inf") and 0.0 < oracle_us < float("inf")
    quality = oracle_us / best_us if usable else 0.0
    error = None if usable else "no valid rung measured"
    return E4Row(kernel, backend, strategy, result.best, best_us, quality, result.ledger.measurements,
                 result.ledger.tokens, usable, error)


def cost_quality_table(rows: Sequence[E4Row]) -> Dict[str, Dict[str, float]]:
    """The E4 figure: strategy -> mean quality and mean measurements over the valid rows. The C4 claim is
    read here -- an agent whose ``quality`` is near 1.0 at a fraction of ``measurements``."""
    table: Dict[str, Dict[str, float]] = {}
    for strategy in sorted({r.strategy for r in rows}):
        valid = [r for r in rows if r.strategy == strategy and r.ok]
        if not valid:
            continue
        table[strategy] = {
            "quality": sum(r.quality for r in valid) / len(valid),
            "measurements": sum(r.measurements for r in valid) / len(valid),
            "tokens": sum(r.tokens for r in valid) / len(valid),
            "runs": float(len(valid)),
        }
    return table


def savings_vs_oracle(rows: Sequence[E4Row]) -> Dict[str, float]:
    """Fraction of the oracle's measurements each strategy spent, over runs where BOTH completed. 0.4 means
    it reached its quality for 40% of the exhaustive cost; the oracle itself is 1.0 by definition."""
    per_run: Dict[str, List[float]] = {}
    oracle_cost = {(r.kernel, r.backend): r.measurements for r in rows if r.strategy == ORACLE and r.ok}
    for r in rows:
        cost = oracle_cost.get((r.kernel, r.backend))
        if r.ok and cost:
            per_run.setdefault(r.strategy, []).append(r.measurements / cost)
    return {name: sum(v) / len(v) for name, v in per_run.items()}
