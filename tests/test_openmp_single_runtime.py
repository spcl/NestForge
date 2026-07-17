"""ONE OpenMP runtime, globally, across every compiler and every lane -- the PARALLEL.md contract.

Every node library and the driver must link the SAME OpenMP runtime, so that libraries built by
DIFFERENT compilers share one runtime and ONE thread pool. Left to a bare ``-fopenmp`` each family
links its own default (gcc -> libgomp, clang -> libomp, icx -> libiomp5), which silently puts TWO
runtimes with two thread pools in one process as soon as a sweep spans gcc and clang.

These tests assert on the ARTIFACT, not on the flag list: ``readelf -d`` over a real ``.so`` says which
runtime actually landed in DT_NEEDED. That distinction is the whole point -- the violation these tests
were written for was invisible to every flag-level assertion, because each family's flags looked
perfectly correct in isolation. Only linking a cell from each compiler and comparing the results shows
the two runtimes.

Nothing here RUNS a kernel: linking is what selects a runtime, so compiling is the whole experiment.
That also keeps the file free of the libgomp fork hazard (an OpenMP region in this process would
poison every later ``run_isolated`` fork).
"""
import shutil
import subprocess
import pytest

from nestforge.build import OPENMP_RUNTIMES, compiler_family
from nestforge.perf import flags

#: A minimal nest with an OpenMP region: enough to make the compiler link a runtime, which is all that
#: is under test. ``omp-emit`` compiles the pragma as written; ``auto-par`` re-derives it.
OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

#: A SECOND, differently-shaped nest. A single kernel could link one runtime by luck of its shape; the
#: contract is about every node library in the program, so the matrix runs two unrelated nests (an
#: elementwise map and a reduction, which lower to different OpenMP constructs -- ``parallel for`` vs
#: ``parallel for reduction``).
OMP_SRC_REDUCE = """#include <omp.h>
double kern2(const double *a, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i] * 2.0;
  return s;
}
"""

#: The OpenMP runtimes a linked object can name, by DT_NEEDED soname stem.
OMP_SONAMES = ("libgomp", "libomp", "libiomp5", "libnvomp")

#: (family label used by flags.py, C compiler exe). The families nest-forge sweeps.
FAMILIES = (("gnu", "gcc"), ("llvm", "clang"), ("intel", "icx"), ("nvidia", "nvc"))


def linked_openmp_runtimes(so):
    """The OpenMP runtimes in ``so``'s DT_NEEDED, as soname stems.

    DT_NEEDED records the SONAME of what the linker RESOLVED, which is not always the ``-l`` name asked
    for: distros ship ``libiomp5.so`` as a symlink onto LLVM's ``libomp.so`` (they are ABI-compatible),
    so requesting libiomp5 legitimately yields a ``libomp.so.5`` entry. Hence the invariant tested here
    is "exactly ONE runtime, and the same one for every compiler" rather than "the name asked for".
    """
    out = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    return {name for name in OMP_SONAMES if f"[{name}.so" in out}


def available_families():
    """The (family, compiler) pairs whose C compiler exists here. Never empty: gcc or clang is present
    on any box that can build a nest at all, and the CI runner has both."""
    return [(fam, cc) for fam, cc in FAMILIES if shutil.which(cc)]


def build_cell(tmp_path, family, compiler, mode, runtime, src=OMP_SRC, tag="k"):
    """Link one sweep cell exactly as a driver does -- ``[exe, *lane_flags, src, -o, so]``, flags BEFORE
    the source. That ordering is load-bearing: it is why a plain ``--as-needed -lomp`` is dropped as
    unused (nothing is undefined yet) and gcc's implicit trailing ``-lgomp`` wins instead. A test that
    put the source first would link the right runtime and prove nothing.

    Returns ``(runtimes, skip_reason)``; exactly one is None.
    """
    f, reason = flags.lane_flags(family, "default-fp", "default", mode, "c", 2, compiler=compiler, openmp=runtime)
    if f is None:
        return None, reason
    csrc = tmp_path / f"{tag}.c"
    csrc.write_text(src)
    so = tmp_path / f"{tag}_{compiler}_{mode}_{runtime.name}.so"
    proc = subprocess.run([compiler, *f, str(csrc), "-o", str(so)], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"{compiler} {mode} {runtime.name} failed to link:\n{proc.stderr[-1500:]}")
    return linked_openmp_runtimes(so), None


# --- the contract ------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omp-emit", "auto-par"])
def test_every_compiler_links_the_same_single_runtime(tmp_path, mode):
    """THE contract: across every available compiler, a cell links exactly ONE OpenMP runtime, and it is
    the SAME one for all of them.

    Before the fix this failed with {'libgomp', 'libomp'}: gcc's bare -fopenmp linked libgomp while
    clang's linked libomp, so a sweep spanning both put two runtimes and two thread pools in one
    process -- the dual-runtime oversubscription OpenMPRuntime exists to prevent.
    """
    seen = {}
    for family, cc in available_families():
        rts, reason = build_cell(tmp_path, family, cc, mode, flags.DEFAULT_OPENMP_RUNTIME, tag="a")
        if rts is None:
            continue  # a family that cannot link the mandated runtime is a recorded skip, not a failure
        assert len(rts) == 1, f"{cc} {mode} linked {len(rts)} OpenMP runtimes ({sorted(rts)}), must be exactly 1"
        seen[cc] = rts
    assert seen, "no compiler could link the default runtime -- the matrix would be vacuous"
    distinct = set().union(*seen.values())
    assert len(distinct) == 1, (f"the single-runtime contract is violated across compilers: {seen} -- one process "
                                f"loading these node libraries would hold {len(distinct)} runtimes + thread pools")


def test_two_different_nests_from_two_compilers_share_one_runtime(tmp_path):
    """The mixed-compiler/single-runtime pair: two UNRELATED nests (elementwise map + reduction, which
    lower to different OpenMP constructs), each built by a different compiler, as they would be when
    linked into one program. The union over the pair must still be one runtime -- that is what makes
    them safe to load together."""
    fams = available_families()
    if len(fams) < 2:
        pytest.skip(f"needs two compiler families, found {[c for _, c in fams]}")
    (fam_a, cc_a), (fam_b, cc_b) = fams[0], fams[1]
    union, built = set(), {}
    for (fam, cc), src, tag in (((fam_a, cc_a), OMP_SRC, "map"), ((fam_b, cc_b), OMP_SRC_REDUCE, "red")):
        rts, reason = build_cell(tmp_path, fam, cc, "omp-emit", flags.DEFAULT_OPENMP_RUNTIME, src=src, tag=tag)
        if rts is None:
            pytest.skip(f"{cc} cannot link the default runtime: {reason}")
        built[f"{cc}:{tag}"] = sorted(rts)
        union |= rts
    assert len(union) == 1, f"two node libraries, two compilers, {len(union)} runtimes: {built}"


@pytest.mark.parametrize("runtime_name", sorted(OPENMP_RUNTIMES))
def test_the_global_runtime_is_choosable_and_prunes_what_cannot_link_it(tmp_path, runtime_name):
    """The runtime is a KNOB, not a constant: every entry in OPENMP_RUNTIMES can be selected, and for a
    given choice each compiler either links exactly that one runtime or is pruned with a reason. A
    compiler must never silently fall back to its own default -- that is the bug, expressed as an axis.

    Pruning is not a formality: libgomp is gomp-ABI only, so clang (which emits __kmpc_*) cannot link it
    and libgomp can never be the GLOBAL runtime of a gcc+clang sweep. That asymmetry is the reason the
    default is libomp, which both families link.
    """
    runtime = OPENMP_RUNTIMES[runtime_name]
    distinct, decided = set(), {}
    for family, cc in available_families():
        rts, reason = build_cell(tmp_path, family, cc, "omp-emit", runtime, tag="c")
        if rts is None:
            assert reason, f"{cc} was pruned for {runtime_name} with no reason recorded"
            decided[cc] = f"skip: {reason}"
            continue
        assert len(rts) == 1, f"{cc} linked {sorted(rts)} for {runtime_name}; must be exactly 1"
        decided[cc] = sorted(rts)
        distinct |= rts
    assert decided, "no compiler was even considered"
    assert len(distinct) <= 1, f"{runtime_name} produced {len(distinct)} distinct runtimes: {decided}"


def test_libgomp_is_pruned_for_llvm_but_kept_for_gnu():
    """The pruned matrix is CORRECT, not merely non-empty -- asserted on the compatibility rules rather
    than on this box's toolchain, so it means the same everywhere.

    libgomp implements only the GOMP_* ABI; clang/flang/icx emit __kmpc_*. So gnu keeps it and every
    kmpc family drops it. Conversely libomp carries a GOMP-compat layer, so it serves BOTH -- which is
    exactly why it is the default global runtime.
    """
    libgomp, libomp = OPENMP_RUNTIMES["libgomp"], OPENMP_RUNTIMES["libomp"]
    assert libgomp.compatible("gcc") and not libgomp.compatible("clang")
    assert libomp.compatible("gcc") and libomp.compatible("clang")
    # ... and the flag axis honours it: the reason is recorded, not swallowed.
    f, reason = flags.lane_flags("llvm", "default-fp", "default", "omp-emit", "c", 2, compiler="clang", openmp=libgomp)
    assert f is None and reason and "libgomp" in reason
    # the native-runtime-only families accept only their own, whatever the global choice says
    assert not libomp.compatible("nvc") and OPENMP_RUNTIMES["libnvomp"].compatible("nvc")
    assert not libomp.compatible("icc") and OPENMP_RUNTIMES["libiomp5"].compatible("icc")
    assert compiler_family("icx") == "llvm" and libomp.compatible("icx")  # icx is clang-based: name-selects libomp


def test_lane_flags_names_the_runtime_rather_than_leaving_it_to_the_compiler_default():
    """The mechanism, per family, without needing the compiler installed (pure composition).

    gnu has no ``-fopenmp=<lib>``, so it must pin the runtime at LINK; llvm selects by name. The gnu
    spelling is push-state/--no-as-needed/pop-state and not a plain ``-l``: these flags precede the
    source, where nothing is undefined yet, so ``--as-needed -lomp`` would be dropped as unused and the
    driver's trailing ``-lgomp`` would win. --no-as-needed alone would link BOTH.
    """
    libomp = OPENMP_RUNTIMES["libomp"]
    gnu, _ = flags.openmp_runtime_flags("gcc", "gnu", libomp)
    assert any("--push-state,--no-as-needed,-lomp,--pop-state" in f for f in gnu), gnu
    llvm, _ = flags.openmp_runtime_flags("clang", "llvm", libomp)
    assert "-fopenmp=libomp" in llvm, llvm
    # no compiler to ask -> pure composition, no flags invented
    assert flags.openmp_runtime_flags(None, "gnu", libomp) == ([], None)
