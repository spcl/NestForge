# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The duplicate-variant key: two variants share it exactly when they are the same build."""

import shutil
import subprocess
from pathlib import Path

import pytest

from nestforge.build import flags as flags_mod
from nestforge.build.flags import FP_LEVELS
from nestforge.build import dedup
from nestforge.build.dedup import asm_bodies, collapse, collapse_notes, parse_disassembly, variant_key
from nestforge.build.sdfg import BuildOptions, build_archive
from nestforge.build.toolchain import compiler_family, needed_libraries

SYMBOL = "k_fp64"
#: A transcendental, in a loop the back end will vectorize, for the needed_libraries link-axis test.
SIN_KERNEL = f"""#include <math.h>
extern "C" void {SYMBOL}(double *__restrict__ a, const double *__restrict__ b, int n) {{
  for (int i = 0; i < n; ++i) a[i] = sin(b[i]);
}}
"""
#: A reduction: reassociation is exactly what separates fast-math from the strict rungs.
SUM_KERNEL = f"""extern "C" void {SYMBOL}(double *__restrict__ a, const double *__restrict__ b, int n) {{
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += b[i] * b[i];
  a[0] = s;
}}
"""

assert shutil.which("g++") is not None, "g++ not on PATH (setup_apt.sh installs it)"
assert shutil.which("clang++") is not None, "clang++ not on PATH (setup_apt.sh installs it)"
assert shutil.which("objdump") is not None, "objdump not on PATH (setup_apt.sh: binutils)"


def build(tmp_path: Path, source: str, fp_mode: str, tag: str = "v", compiler: str = "g++") -> Path:
    """Build ``source`` as a phase 5 variant is built; returns the object the key reads."""
    src = tmp_path / f"{tag}.cpp"
    src.write_text(source)
    family = compiler_family(compiler)
    composed = [*flags_mod.BASE_FLAGS, *flags_mod.fp_flags(family, fp_mode), *flags_mod.cost_flags(family, "default")]
    out = tmp_path / tag
    opts = BuildOptions(compiler=compiler, flags=composed)
    build_archive([src], out, out / f"lib{tag}.a", out / f"lib{tag}.so", opts)
    return out / f"{tag}.o"


# the key


def test_the_key_separates_fp_rungs_of_one_source(tmp_path):
    """Compile flags never reach the source, so only the object shows that reassociation changed the code."""
    strict = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="strict")
    fast = build(tmp_path, SUM_KERNEL, "fast-math", tag="fast")
    groups = collapse({"strict": variant_key(strict, SYMBOL) or "", "fast": variant_key(fast, SYMBOL) or ""})
    assert len(groups) == 2, groups


#: One objdump body, verbatim shape: an immediate, a rip-relative load with a relocation comment, and a
#: branch whose target objdump already annotates with the symbol.
DISASM = """
0000000000000000 <k_fp64>:
   0:\tendbr64
   4:\tadd    $0x2,%eax
   7:\tmovsd  0x0(%rip),%xmm1        # f <k_fp64+0xf>
   c:\tcall   1090 <sin@plt>
  11:\tret
"""


def test_parsing_strips_the_address_column_the_reloc_comment_and_the_branch_slot():
    """Each of the three is layout, not content: the address column shifts with position, the reloc
    comment repeats it, and a PLT slot moves between links of identical code."""
    body = parse_disassembly(DISASM)["k_fp64"]
    assert body.splitlines() == ["endbr64", "add    $0x2,%eax", "movsd  0x0(%rip),%xmm1", "call   <sin@plt>"] + ["ret"]


def test_parsing_keeps_an_immediate_that_only_differs_in_value():
    """The exact risk :data:`BRANCH_TARGET` creates -- stripping a branch address must not strip an
    IMMEDIATE. Asserted on the normalizer directly: two objects differing by one operand are hard to get
    out of a real compiler, which is what made an end-to-end version of this test pass vacuously."""
    two = parse_disassembly(DISASM)["k_fp64"]
    three = parse_disassembly(DISASM.replace("$0x2", "$0x3"))["k_fp64"]
    assert two != three, "an operand is content; erasing it collapses two genuinely different builds"


def test_parsing_keeps_a_rip_offset_that_only_differs_in_value():
    """Same argument one step out: the reloc COMMENT is dropped, the addressing operand is not."""
    a = parse_disassembly(DISASM)["k_fp64"]
    b = parse_disassembly(DISASM.replace("0x0(%rip)", "0x8(%rip)"))["k_fp64"]
    assert a != b


def test_asm_bodies_strips_addresses_but_keeps_the_instructions(tmp_path):
    """Addresses shift with layout and would defeat the match; the mnemonics are the content."""
    obj = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="body")
    body = asm_bodies(obj)[SYMBOL]
    assert body and "ret" in body
    assert not any(line.strip().startswith(("0x", "00000")) for line in body.splitlines())


def test_naming_an_absent_symbol_is_an_error_not_a_silent_whole_object_key(tmp_path):
    """Falling back to the whole object would key on init/exit boilerplate and quietly answer a different
    question than the caller asked."""
    obj = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="missing")
    with pytest.raises(LookupError, match="nope"):
        variant_key(obj, "nope")


def test_an_object_with_no_disassembly_has_no_key_rather_than_the_key_of_nothing(tmp_path):
    """Two unreadable objects both hashing the empty string would collapse into one measured variant."""
    empty = tmp_path / "not_an_object.o"
    empty.write_text("")
    assert variant_key(empty, SYMBOL) is None


def test_needed_libraries_reads_the_link_axis_the_object_key_misses(tmp_path):
    """One object, two links: keying a ``.so`` means composing the two -- same code plus a different
    resolver is still a different variant to measure."""
    obj = build(tmp_path, SIN_KERNEL, "fast-math", tag="need")
    bare, withlib = tmp_path / "libbare.so", tmp_path / "libwith.so"
    subprocess.run(["gcc", "-shared", str(obj), "-o", str(bare)], check=True, capture_output=True)
    subprocess.run(
        ["gcc", "-shared", str(obj), "-Wl,--no-as-needed", "-lmvec", "-o", str(withlib)],
        check=True,
        capture_output=True,
    )
    assert asm_bodies(bare)[SYMBOL] == asm_bodies(withlib)[SYMBOL], "the code really is the same"
    assert any("mvec" in soname for soname in needed_libraries(withlib))
    assert variant_key(bare, SYMBOL) != variant_key(withlib, SYMBOL)


#: No FMA to contract, no math call to relax, nothing to reassociate: the fp ladder cannot reach it.
ADD_KERNEL = f"""extern "C" void {SYMBOL}(double *__restrict__ c, const double *__restrict__ a, const double *__restrict__ b,
    long n) {{
  for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
}}
"""


def test_the_pruner_collapses_fp_rungs_a_kernel_cannot_tell_apart(tmp_path):
    """Every FP mode compiles this kernel to the same object, so the sweep measures it once."""
    keys = {rung: variant_key(build(tmp_path, ADD_KERNEL, rung, tag=rung), SYMBOL) for rung in FP_LEVELS}
    readable = {rung: key for rung, key in keys.items() if key is not None}
    assert readable == keys, keys
    groups = collapse(readable)
    assert len(groups) < len(FP_LEVELS), f"no rung collapsed, so the pruner saves nothing here: {groups}"
    assert collapse_notes(groups)


# grouping


def test_collapse_groups_by_key_and_keeps_the_first_as_representative():
    groups = collapse({"a": "k1", "b": "k2", "c": "k1"})
    assert groups == {"k1": ["a", "c"], "k2": ["b"]}


def test_collapse_notes_report_what_was_collapsed():
    """A silent collapse reads exactly like a sweep that covered everything."""
    assert collapse_notes(collapse({"a": "k1", "b": "k2", "c": "k1"})) == ["a == c"]


def test_collapse_notes_are_empty_when_nothing_collapsed():
    assert collapse_notes(collapse({"a": "k1", "b": "k2"})) == []


def test_an_artifact_whose_link_cannot_be_read_has_no_key(monkeypatch, tmp_path):
    """A failing ``readelf`` must mean measuring the variant, not aborting the sweep."""

    def unreadable(shared: Path) -> list[str]:
        raise subprocess.CalledProcessError(1, ["readelf", "-d", str(shared)])

    monkeypatch.setattr(dedup, "asm_bodies", lambda obj: {SYMBOL: "ret"})
    monkeypatch.setattr(dedup, "needed_libraries", unreadable)

    assert variant_key(tmp_path / "lib.so", SYMBOL) is None
