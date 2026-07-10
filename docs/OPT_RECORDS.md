# Parsing compiler optimization records — GCC, LLVM/Clang, Intel, NVIDIA HPC

nest-forge's **predictive-compiling** mode (see `PREDICTIVE.md` §2A) ranks compiler×flag combinations
*without running the binary*: compile-only, parse each compiler's optimization report, score each loop
by whether it vectorized, at what width, with what interleave/unroll, and why it was refused, then only
*profile* the top-k. nest-forge's edge over a generic model is that it **extracted the nest**, so it
knows the iteration space (trip counts from the size symbols) — a cheap explainable score
`Σ_loops (vector_width · trip_count / latency)`, penalized by missed-vec / spills / remainder loops,
ranks compilers before any run.

This document is the implementation reference for emitting and machine-parsing those reports across the
four toolchains. LLVM/Clang and Intel icx/ifx offer the richest **structured** record (YAML /
bitstream); GCC has a structured record too (gzip-JSON, GCC 9+, experimental) but its stable contract is
the **text** `-fopt-info` lines; NVIDIA and classic Intel are text only and need line regexes.

## Normalized schema

Parse everything into one record so the ranker is compiler-agnostic:

```
{ file, line, col, function, loop_id, pass,
  status: "passed" | "missed" | "analysis",
  vector_width?, interleave?, unroll?, reason?, estimated_speedup?, raw }
```

Reliably available from **all four**: `file`, `line`, vectorized-yes/no, and (for successes) vector
width. Compiler-specific: Intel's `estimated potential speedup` (`#15478`) and scalar/vector cost
(`#15476`/`#15477`); LLVM's structured `VectorizationFactor`/`InterleaveCount` args; NVIDIA's
`function:`-grouped unroll/prefetch notes. `col` is present for GCC/LLVM/Intel but **not** NVIDIA (line
only).

## GCC (gcc / g++ / gfortran) — text (stable) + gzip-JSON (GCC 9+)

Two interfaces: the stable **text** `-fopt-info` lines, and a structured **gzip-JSON** record
(`-fsave-optimization-record`, GCC 9+, self-described as experimental).

**Text.** `-fopt-info[-<type>][-<group>][=<file>]` — order-free, so `-fopt-info-vec-missed` ==
`-fopt-info-missed-vec`. Default (bare `-fopt-info`) is `optimized-optall`. `<type>` ∈
`{optimized, missed, note, all}`; `<group>` ∈ `{vec, loop, inline, ipa, omp, optall}`. Default sink is
**stderr**; `=<file>` redirects (there is **no** `=stderr` keyword — stderr is just the no-file default;
only one `-fopt-info` may carry a filename). Useful invocations: `-fopt-info-vec` (were vectorized),
`-fopt-info-vec-missed` (were not, with reason), `-fopt-info-vec-all` (both + notes),
`-fopt-info-all` (everything).

Verbatim output shape is `file:line:col: <kind>: <message>`:

```
main.c:4:26: optimized: loop vectorized using 16 byte vectors
sum.c:4:4: optimized: loop vectorized using 32 byte vectors
test.c:11:31: optimized: loop vectorized using 16 byte vectors and unroll factor 8
no-vfa-vect-102.c:24:3: missed: couldn't vectorize loop
foo.c:7:3: note: not vectorized, possible dependence between data-refs
```

Parse regex: `^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<kind>optimized|missed|note):\s+(?P<msg>.*)$`,
then dispatch on `kind` (optimized→passed, missed→missed, note→analysis) and mine `msg` for
`(\d+) byte vectors` and `unroll factor (\d+)`. (`optimized:`/`missed:` are not standard diagnostic
severities, so generic gcc-diagnostic parsers choke — use this dedicated regex.)

**gzip-JSON.** `-fsave-optimization-record` writes `SRCFILE.opt-record.json.gz`, "roughly equivalent to
a machine-readable version of `-fopt-info-all`" (GCC 9, 2019; the manual flags the format "experimental…
subject to change"). Read with stdlib only: `json.load(gzip.open(path))`. The file is a JSON array of
`[metadata, passes[], records[]]`; each record carries `kind` (optimized/missed/note), `message` (an
array mixing strings and `{expr:…}` objects), `pass` (an id into `passes[]`, so unlike the text this
**names the pass**), `count` (profile count), `location {file,line,column}`, `function`, and
`inlining_chain`. Use it when you want the pass name / inlining chain without prose-parsing; otherwise
the text lines are the stable contract.

Facts for the parser:
- Nothing is reported without the pass running: the vectorizer needs `-ftree-vectorize` (on at `-O3`
  always, and at `-O2` since **GCC 12** with the very-cheap cost model; before GCC 12, `-O2` did not
  auto-vectorize). The unroller needs `-funroll-loops` (off by default even at `-O3`).
- **GCC text lines do not name the pass** — recover it from the requested `-fopt-info-<group>`, from
  keyword-matching the message, or from the JSON record. LLVM always names the pass; GCC text never does.
- `-march=native` changes which loops vectorize and the reported width (`16`/`32`/`64` byte vectors ↔
  SSE/AVX/AVX-512). Text source locations do **not** require `-g` (unlike LLVM).
- GCC reports **byte** width ("32 byte vectors"); LLVM reports element **lanes** (`VectorizationFactor:
  4`) and Intel reports `vector length N` (lanes) — normalize GCC bytes→lanes with the dtype size before
  comparing. The heavyweight IR dumps (`-fdump-tree-vect-details`) are voluminous and unstable — not
  worth parsing for ranking.

## LLVM / Clang / flang — structured (the richest path)

Two channels share one data model (`llvm::remarks`): **stderr diagnostics** (`-Rpass*`) and a
**serialized record file** (`-fsave-optimization-record`). Same `Pass`/`Name`/`Args` tokens in both.

Flags (clang driver):
- `-fsave-optimization-record` → writes `<output-basename>.opt.yaml`, one per TU. `=yaml` (default) or
  `=bitstream`.
- `-foptimization-record-file=<path>` (implies save), `-foptimization-record-passes=<regex>` (filter).
- `-Rpass=<regex>`, `-Rpass-missed=<regex>`, `-Rpass-analysis=<regex>` — stderr remarks, matched
  against **pass** names, e.g. `-Rpass=loop-vectorize`. `-Rpass=.*` for every pass.
- **Require `-gline-tables-only` (`-g1`)** so `DebugLoc` (File/Line/Column) is populated — remarks are
  translated from debug annotations.
- `opt`/`llc`/LTO equivalents: `-pass-remarks[-missed|-analysis]=<regex>` (stderr),
  `-pass-remarks-output=<file>`, `-pass-remarks-format=<yaml|bitstream>`.
- **flang** supports the identical flags — treat its output exactly like clang's.

Stderr shape (verbatim): `a.cpp:5:5: remark: vectorized loop (vectorization width: 4, interleaved count:
2) [-Rpass=loop-vectorize]`; failures `a.cpp:6:5: remark: loop not vectorized [-Rpass-missed=loop-vectorize]`
and analysis `... loop not vectorized: could not determine number of loop iterations
[-Rpass-analysis=loop-vectorize]`. **The success wording drifted** between LLVM versions (old:
`vectorization factor: 4, unrolling interleave factor: 2`); prefer the YAML `Args` keys, which are
stable.

YAML schema — a stream of documents, one per remark, `---` … `...`:

```yaml
--- !Passed
Pass:            loop-vectorize
Name:            Vectorized
DebugLoc:        { File: foo.c, Line: 21, Column: 3 }
Function:        Test
Args:
  - String:          'vectorized '
  - String:          ''
  - String:          'loop (vectorization width: '
  - VectorizationFactor: '4'
  - String:          ', interleaved count: '
  - InterleaveCount:   '2'
  - String:          )
...
```

- **Document tags (remark type):** `!Passed`, `!Missed`, `!Analysis`, `!AnalysisFPCommute`,
  `!AnalysisAliasing`, `!Failure`. (`!AnalysisFPCommute` is precisely the FP-reassociation-legality
  analysis — relevant to `fp_risk`.) `!Failure` is emitted by pass **`transform-warning`** (not
  `loop-vectorize`) when an explicit `#pragma` couldn't be honored — filtering on `Pass:
  loop-vectorize` alone misses it.
- **Key on `(Pass, Name)`.** Pull structured `Args` (`VectorizationFactor`, `InterleaveCount`, unroll
  `UnrollCount`, inline `Cost`/`Threshold`) as features — not the version-unstable prose. A `String:`
  arg is a message fragment; a `Key: value` arg is a machine value (`ore::NV(...)` in the pass).
- **`Name` catalog** (from LLVM sources): success `Vectorized` (args `VectorizationFactor`,
  `InterleaveCount`), `Interleaved` (scalar-VF, arg `InterleaveCount`); miss/cost `MissedDetails`,
  `VectorizationNotBeneficial`, `InvalidCost`; legality failures (all prefixed `loop not vectorized: `)
  `CFGNotUnderstood`, `NoInductionVariable`, `UnsupportedUncountableLoop`,
  `LoopContainsUnsupportedSwitch`, `CantVectorizeLibcall`, `NonReductionValueUsedOutsideLoop`, … ;
  `transform-warning`/`FailedRequestedVectorization`. SLP: `VectorizedList`, `StoresVectorized`,
  `VectorizedHorizontalReduction`. Inline: `NoDefinition` (+ `Callee`/`Caller` args).

Parsing tooling (reuse, don't reinvent):
- **`optrecord.py`** (`llvm/tools/opt-viewer/`) is the reference YAML parser — PyYAML with
  `CLoader`/`CSafeLoader` (install **libYAML** for speed); registers the six tags as classes. Model
  nest-forge's parser on it; the hardened fork **OfekShilon/optview2** filters to actionable misses.
- **`llvm-opt-report file.opt.yaml`** reduces the YAML to source-annotated `I`/`U`/`V` markers
  (`V4,1` = vectorized width 4 interleave 1; `U16` = unrolled ×16; `I` = inlined) — a ready-made
  oracle for exactly the facts the ranker wants.
- **`llvm-remarkutil bitstream2yaml`** converts the compact bitstream form; `libRemarks` C API
  (`LLVMRemarkParserGetNext`) reads either format without PyYAML.

Recommendation: emit **bitstream** at scale (small, fast; magic `RMRK`, string-table dedup), convert per
file with `llvm-remarkutil bitstream2yaml` or bind `libRemarks`, parse as a multi-document YAML stream
keyed on `(Pass, Name)`.

## Intel (icx / icpx / ifx, and classic icc / ifort)

Flags: `-qopt-report[=n]` (`/Qopt-report` on Windows). Levels: **classic 0–5** (2 default, 5 = data-
dependence detail); **icx/ifx (LLVM-based) 0–3** (3 = max; no 4/5). `-qopt-report-phase=vec,loop,...`
(default all; `vec` is the highest-value phase), `-qopt-report-file=<f|stderr|stdout>`,
`-qopt-report-format=text|vs`. Default sink: **one `.optrpt` file per source** (text, line-oriented —
*not* JSON/XML).

Text `.optrpt` structure — a `LOOP BEGIN at file(line,col)` … `LOOP END` bracket stack under
`Begin optimization report for: <FUNC>` / `Report from: <phase> [tag]`, with `remark #NNNNN:` lines
(bare or `file(line,col):remark #NNNNN:`). Verbatim:

```
LOOP BEGIN at test.cpp(3,2)
   test.cpp(3,2):remark #15305: vectorization support: vector length 2
   test.cpp(3,2):remark #15300: LOOP WAS VECTORIZED
   test.cpp(3,2):remark #15475: --- begin vector cost summary ---
   test.cpp(3,2):remark #15476: scalar cost: 8
   test.cpp(3,2):remark #15477: vector cost: 3.500
   test.cpp(3,2):remark #15478: estimated potential speedup: 2.250
   test.cpp(3,2):remark #15488: --- end vector cost summary ---
LOOP END
LOOP BEGIN at novec.f90(4,3)
   remark #15344: loop was not vectorized: vector dependence prevents vectorization
   remark #15346: vector dependence: assumed FLOW dependence between y(i) and y(i-1)
LOOP END
```

Remark-number bands: `#10xxx` driver/info, `#15xxx` vectorizer, `#25xxx` loop-nest/memory. The
**single most predictive signals** live in the cost summary: `#15476 scalar cost`, `#15477 vector cost`,
**`#15478 estimated potential speedup`** — Intel is the only compiler that hands you a speedup estimate
directly. Success `#15300 LOOP WAS VECTORIZED` / `#15301` (PARTIAL/REMAINDER/OpenMP-SIMD variants);
width `#15305 vector length N`; unroll `#15399`; memref quality `#15450/#15451` (unaligned unit stride),
`#15458/#15459` (gather/scatter); failures `#15344/#15346` (dependence), `#15335` (inefficient),
`#15331` (precise-FP model), `#15382` (call); loop transforms `#25426` (distributed), `#25436/#25438`
(unroll), `#25442` (blocked), `#25444` (interchange), `#25015` (max trip count estimate).

Parse regex:
`^\s*(?:(?P<loc>[^:]+\(\d+,\s*\d+\))\s*:)?\s*remark #(?P<num>\d+):\s*(?P<msg>.*)$`, plus a
`LOOP BEGIN/END` stack for nest depth. Numbers are stable across versions; the trailing text carries the
operands.

**icx/ifx caveat:** LLVM-based, so `-qopt-report` also (historically) emitted an **LLVM `.opt.yaml`**
(the same schema as the LLVM section) — but **since oneAPI 2025.0 the YAML is no longer auto-emitted**;
request it with `-fsave-optimization-record`. The icx **text** `.optrpt` uses the same `remark #NNNNN`
taxonomy but a **sparser subset** (fewer `#15475–#15488` cost lines). So for icx, prefer the LLVM YAML
path (`-fsave-optimization-record`, parsed as in the LLVM section) and treat "cost block absent" as a
distinct feature rather than zero.

## NVIDIA HPC SDK (nvc / nvc++ / nvfortran, formerly PGI)

Flags: `-Minfo[=<group>]` (what *was* optimized) and `-Mneginfo[=<group>]` (why *not*), groups
`{all, vect, loop, opt, unroll, inline, mp, par, accel, stdpar, ...}` — note the spelling **`vect`**, not
`vec`. **No file option, no structured format** — plain text to **stderr**; capture with `2>report.txt`.
`-Minfo` is purely a reporting flag; what it prints depends on `-O`/`-fast`/`-acc`/`-mp`.

Output is a two-level indent: a **`function:`** header flush at column 0, then indented
**`<line>, <message>`** lines (the leading integer is a **source line, never a column**), with
no-number continuation lines binding to the previous line. Verbatim:

```
vector_op:
       4, Loop unrolled 16 times
       9, Loop not vectorized/parallelized: contains call
loop:
      18, Generated 2 alternate versions of the loop
          Generated vector simd code for the loop
          FMA (fused multiply-add) instruction(s) generated
```

Parse (state machine): `HEADER = ^(\S.*):\s*$` (col-0, ends `:`), `MSG = ^(\s+)(\d+),\s?(.*)$` (line +
message), `CONT = ^\s+(\D.*\S)\s*$` (continuation, no number → attach to current line). Positive
matchers: `Generated vector simd code`, `Generated vector sse code` (older), `Loop unrolled N times`,
`FMA ... generated`, `Generated N alternate versions`, `Accelerator kernel generated` /
`Generating (Tesla|NVIDIA GPU) code` (match both spellings). Negative (also appear under `-Minfo`, not
only `-Mneginfo`): any `not vectorized|not parallelized|not fused|prevents|dependence` substring, e.g.
`Loop not vectorized/parallelized: contains call`, `Loop not vectorized: mixed data types`,
`Complex loop carried dependence of <var> prevents parallelization`. Always compile with
`-Minfo=all -Mneginfo=all`. No `estimated speedup` and no `col` — line only.

## Synthesis

| | GCC | LLVM/Clang/flang | Intel icx/ifx | Intel classic | NVIDIA nvhpc |
|---|---|---|---|---|---|
| structured format | gzip-JSON (GCC 9+, experimental) | **YAML + bitstream** | LLVM YAML (opt-in 2025+) | no (`.optrpt` text) | no (stderr text) |
| enable | `-fopt-info-vec[-missed]` | `-fsave-optimization-record` / `-Rpass` | `-qopt-report=3` / `-fsave-optimization-record` | `-qopt-report=5` | `-Minfo=all -Mneginfo=all` |
| sink | stderr / `=file` | `.opt.yaml` per TU / stderr | `.optrpt` per src | `.optrpt` per src | stderr |
| source loc | file:line:col | File/Line/Column (needs `-g1`) | file(line,col) | file(line,col) | line only (no col) |
| width field | bytes ("32 byte") | `VectorizationFactor` (elems) | `vector length N` | `vector length N` | `vector simd` (no width) |
| speedup estimate | — | — | — (sparse) | **`#15478`** | — |
| parse strategy | line regex | YAML load, key `(Pass,Name)` | remark# regex + LOOP stack | remark# regex + LOOP stack | 3-line state machine |

Two parse strategies cover all four: **YAML load** for the structured emitters (LLVM, icx via
`-fsave-optimization-record`) — robust, version-stable, use the structured `Args`; and **line regex**
for the text emitters (GCC `file:line:col: kind:`, Intel classic `remark #NNNNN` + LOOP-stack, NVIDIA
`function:`/`line,` indent). Prefer structured keys over prose everywhere (LLVM prose and Intel's `#15301`
string both drift between versions).

Cross-compiler feature set for the ranker (reliably present everywhere): **vectorized yes/no** and
**source location**. Where available, add: vector **width** (GCC bytes→elements, LLVM
`VectorizationFactor`, Intel `vector length`), **interleave/unroll** (LLVM `InterleaveCount`/`UnrollCount`,
Intel `#15399`/`#25438`, NVIDIA `unrolled N times`), **missed-reason class** (dependence / cost-model /
call / unsupported), and Intel's **`estimated potential speedup`** as a direct label. Combine with
nest-forge's own trip counts:

```
score(compiler, flags) = Σ_loops  I[vectorized] · width_elems · min(trip_count, ∞) / latency_model
                         − penalty(missed_vec) − penalty(spills) − penalty(remainder_loop)
```

Rank by score; profile only the top-k. When Intel's `#15478` is present, use it directly as a
per-loop multiplier; otherwise infer from width×trip.

Existing libraries: LLVM ships `optrecord.py` (the reference YAML model), `opt-viewer.py` /
`llvm-opt-report` (source-annotated), `llvm-remarkutil` (format conversion + `count`/`size-diff`), and
`libRemarks` (C API). There is no equivalent for GCC/NVIDIA/classic-Intel text — those need the regexes
above. Nothing on PyPI parses all four; nest-forge's normalized schema is the unifying layer.
