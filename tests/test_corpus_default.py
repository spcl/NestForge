# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The default kernel corpus is the installed hpcagent_bench ``foundation`` track.

Two things have to hold for that default to be safe, and neither is obvious from the code: a kernel must
keep the SAME key whichever corpus it came from (so ``--only s221`` and every recorded result key survive
the switch), and the metadata hpcagent_bench does not carry must still reach the sizer.
"""
import importlib.util
import sys

from nestforge import tsvc, vendored
from nestforge.corpus import iter_dace_kernels


def test_the_default_corpus_is_foundation():
    """foundation is the corpus that ships WITH the project (submodule or install); the other two are
    scripts inside a DaCe checkout, which a bare clone need not have."""
    import inspect

    assert inspect.signature(tsvc.iter_tsvc_kernels).parameters["corpus"].default == "foundation"
    kernels = tsvc.iter_tsvc_kernels()
    assert kernels and all(k.corpus == "foundation" for k in kernels)


def test_foundation_covers_both_dace_corpora():
    """The switch is only safe because foundation is a superset. Measured, not assumed -- foundation is
    maintained independently and could drift, which would silently shrink every sweep."""
    gap = tsvc.foundation_coverage_gap()
    assert gap == {"tsvc2": [], "tsvc2_5": []}, f"foundation no longer covers: {gap}"


def test_a_kernel_keeps_one_key_across_corpora():
    """hpcagent_bench files the TSVC kernels under a ``tsvc_2_`` stem. Left unnormalised, switching the
    default renames every kernel: `--only s221` selects nothing, and results cannot be compared with any
    run made before the switch."""
    assert tsvc.foundation_key("tsvc_2_s221") == "s221"
    assert tsvc.foundation_key("argmax_value") == "argmax_value"  # a non-TSVC foundation kernel is untouched
    keys = {k.key for k in tsvc.iter_tsvc_kernels()}
    assert "s221" in keys and "tsvc_2_s221" not in keys


def test_de_stemming_collides_with_nothing():
    """Stripping the stem is only sound while no bare foundation kernel is named like a de-stemmed one --
    a collision would silently drop one of the two from the corpus (they key the same dict entry)."""
    names = [k.short_name.rsplit("/", 1)[-1] for k in iter_dace_kernels("foundation")]
    bare = {n for n in names if not n.startswith(tsvc.FOUNDATION_TSVC_STEM)}
    destemmed = {tsvc.foundation_key(n) for n in names if n.startswith(tsvc.FOUNDATION_TSVC_STEM)}
    assert not (bare & destemmed), f"de-stemming collides: {sorted(bare & destemmed)}"


def test_scalar_loop_parameters_survive_the_switch():
    """hpcagent_bench manifests carry problem-size presets but NOT a kernel's scalar loop parameters, and
    `sample_sizes` RAISES on a work-deciding symbol it cannot size. Without the enrichment the 11 kernels
    that have such parameters would drop out of every foundation sweep as skips."""
    by_key = {k.key: k for k in tsvc.iter_tsvc_kernels()}
    assert by_key["s122"].params == {"n1": 1, "n3": 2}
    assert by_key["s162"].params == {"k": 3}


def test_reported_regime_and_tags_survive_the_switch():
    """Both are reported per kernel (`tsvc_full`/`tsvc_arena` write them into every row). Defaulted, all
    245 kernels would read as an untagged 1-D corpus -- wrong data, not a missing field."""
    by_key = {k.key: k for k in tsvc.iter_tsvc_kernels()}
    assert by_key["s1115"].regime == "2d" and "2d" in by_key["s1115"].tags


def test_metadata_enrichment_is_best_effort(monkeypatch):
    """A checkout without DaCe's corpus script must still get every foundation kernel: only the handful
    with scalar parameters may then fail, and they fail LOUDLY at sizing rather than silently vanishing."""

    def no_corpus(*a, **kw):
        raise FileNotFoundError("no performance_regression_jobs in this checkout")

    monkeypatch.setattr(tsvc, "iter_tsvc_kernels", no_corpus)
    assert tsvc.corpus_metadata() == {}


# --- the pinned submodule -----------------------------------------------------------------------------
def test_the_shim_never_overrides_a_resolvable_package():
    """A developer working across both repos must keep getting their own checkout, and re-running the shim
    must not stack a second copy on sys.path. This module cannot even import unless hpcagent_bench already
    resolves, so the answer is False whichever way it resolved -- installed, or via the fallback."""
    assert importlib.util.find_spec("hpcagent_bench") is not None
    before = list(sys.path)
    assert vendored.use_vendored_bench() is False
    assert sys.path == before


def test_the_submodule_is_checked_out():
    """The corpus is the DEFAULT corpus, so a clone without the submodule AND without an install can
    measure nothing at all -- `git submodule update --init` is a setup step, not an optional extra."""
    assert (vendored.VENDORED_BENCH / "hpcagent_bench" / "__init__.py").is_file(), (
        f"{vendored.VENDORED_BENCH} is empty -- run `git submodule update --init --remote`")


def test_the_fallback_fires_when_the_package_is_absent(monkeypatch):
    """Simulate the bare clone: with hpcagent_bench unimportable, the submodule must go on sys.path."""
    monkeypatch.setattr(vendored.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert vendored.use_vendored_bench() is True
    assert str(vendored.VENDORED_BENCH) in sys.path
