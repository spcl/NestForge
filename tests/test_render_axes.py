"""The README configuration-space figure is generated from the live axis constants; this test is what
keeps it honest. If someone adds a value to an axis tuple (a new opt-mode, codegen impl, veclib, ...) and
forgets to regenerate, ``test_readme_figure_is_current`` fails with the fix command."""
import pytest

from nestforge import tsvc
from nestforge.build import CODEGEN_IMPLS
from nestforge.perf import flags, render_axes

pytest.importorskip("optarena")


def test_readme_figure_is_current():
    """The committed README already contains exactly the freshly-generated block -- i.e. no axis changed
    without a regenerate. Fix: python -m nestforge.perf.render_axes --write."""
    assert render_axes.is_fresh(), (
        "README configuration-space figure is stale; run: python -m nestforge.perf.render_axes --write")


def test_figure_lists_every_axis_value():
    """Every value of every live axis tuple appears as a leaf, so the figure can't silently omit an axis
    the arena actually sweeps."""
    diagram = render_axes.mermaid()
    for value in (*tsvc.OPT_MODES, *CODEGEN_IMPLS, *flags.PARALLEL_MODES, *flags.COST_MODELS, *flags.REDUCED_FP_MODES):
        assert f'"{value}"' in diagram, f"axis value {value!r} missing from the generated figure"


def test_lane_cell_counts_are_the_axis_product():
    """Each lane's annotated cell count is the product of its axis sizes -- the figure's headline number."""
    for lane in render_axes.lanes():
        product = 1
        for ax in lane.axes:
            product *= len(ax.values)
        assert lane.cells() == product
        assert f"{lane.cells()} cells" in render_axes.mermaid()


def test_splice_is_idempotent():
    """Regenerating an already-current README is a no-op (splice twice == splice once)."""
    once = render_axes.splice(render_axes.readme_path().read_text())
    twice = render_axes.splice(once)
    assert once == twice
