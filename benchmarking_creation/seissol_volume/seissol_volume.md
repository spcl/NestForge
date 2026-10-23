# seissol_volume (H2 contender, hard, CPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/dense_linear_algebra/seissol_tensor_contraction` (`2666c06`). Written 2026-10-23 from Daint jobs 5127053 (first version, lifted to an `einsum` node), 5127112 (move walk from stage 1, CPU and GPU), 5127336 and 5127372 (route replay), 5127373, 5127388 and 5127389 (stage 3 to 5 builds against the oracle, CPU and GPU), 5127637 (quickstart of this folder). Experiment note: `2026-10-23-h2-contenders` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `entry`, not timed. Route legal and oracle-checked on CPU and GPU at preset S; the gap to the default at preset M is not measured yet.

Versions: NestForge `h2/hard-contenders` (from `ib/on-main` `a823380`; the probes ran on `ib/on-main` `f9081bd`), DaCe `extended` `042413d`, HPCAgent-Bench `2666c06`.

## What this benchmark is

The volume-integral update of SeisSol's ADER-DG scheme (seismic wave propagation): `upd[b, k, p] = sum_{d, l, q} kDivM[d, k, l] * I[b, l, q] * star[d, q, p]`, then `Q += dt * upd`. `b` runs over elements, `k, l` over basis functions (56 for order 6), `p, q` over the 9 elastic quantities, `d` over the 3 dimensions. The contraction is the corpus kernel's einsum; the time-step form (`upd` then `Q += dt * upd`) is how the update is applied. The DaCe program is written output-stationary: one `(b, k, p)` entry, zeroed, accumulated over `d, l, q`. The scheduling problem: on CPU, move the summed loops `d` and `l` outside the `k` and `p` maps, so each `kDivM[d, k, l]` is applied to a whole `(k, p)` block of the element.

## Why it is a good test

- **What the default does instead:** `default_moves()` keeps `map (b, k, p) { for d { for l { map q (reduce) } } }` plus the separate update map. Moves left after it: one `map-fission`, one `subgraph-fission` (job 5127112).
- **What makes it hard:** no interchange is offered from stage 1, only four `interchange-map-map` swaps, a `map-fission` and a `subgraph-fission`. `map-fission` of the `p` map unlocks the `p` pushes; `map-fission` of the `k` map then unlocks the `k` pushes (jobs 5127336, 5127372). The fissions gain nothing alone.
- **The source form matters:** written as one perfect nest (`Q += ...` inside the loops), the frontend lifts it to a single `einsum` library node and stage 2 has no move at all (job 5127053).
- **Tempting wrong moves:** the `interchange-map-map` swaps of `b`, `k`, `p` (offered throughout); stopping after the `p` pushes (3 moves); pushing `b` inward too (it is the parallel axis).

## What the reference is

- **Baseline (CPU and GPU):** stage 1, then `default_moves()`.
- **Reference (CPU):** the 6 moves below, then `finish_moves()`: `map b { init; for d { for l { map k { map p { map q (reduce) } } } } }`, then the update map.
- **GPU:** not decided; the baseline is the expected reference.

Both match the oracle at preset S on CPU (8 OpenMP threads; relative error 1.3e-12, jobs 5127373, 5127388) and on GPU (1.1e-12, job 5127389).

## How it was created

From the corpus einsum, written output-stationary. The first version (perfect nest, job 5127053) had no moves; the second adds the zero init and the separate time-step update, as the ADER-DG step applies it.

## What the solution is

1. `map-fission` on `kernel3_0` (the `p` map): init and sum become two maps.
2. `interchange-map-loop` on (`kernel4_1`, `for5_0`): the `p` sum map goes inside `l`.
3. `interchange-map-loop` on (`kernel5_0`, `for6_0`): and inside `d`.
4. `map-fission` on `kernel2_0` (the `k` map).
5. `interchange-map-loop` on (`kernel3_1`, `for4_0`): the `k` map goes inside `d`.
6. `interchange-map-loop` on (`kernel4_1`, `for5_0`): and inside `l`.
7. `finish_moves()`.

Labels from jobs 5127372 and 5127388; they are not stable across versions.

## How to optimize it

| Decision | What happens | Evidence |
| --- | --- | --- |
| `default_moves()` | one output map, sums inside | 5127112, built 5127373 |
| move 1 only | init split from the sum | built 5127373 |
| moves 1 to 3 | `p` inside `d, l`; `k` outside | built 5127373, 5127389 |
| moves 1 to 6 | `k` and `p` inside `d, l` | built 5127388, 5127389 |
| `interchange-map-map` (`b`, `k`) | swap only | built 5127388 |

## What the gains are

Not measured. Preset M is `NB 40003, NBF 56, NQ 9, ND 3`. Expected (a guess, not measured): on CPU the reference reuses each `kDivM[d, k, l]` across the element's `p` row and walks `I[b, l, :]` contiguously, where the baseline re-reads `kDivM` and `star` for every output entry. To be timed in the batch session.

## CPU and GPU

CPU: the 6-move route. GPU: not decided; the GPU builds of the default, moves 1 to 3 and the route are all correct.

## What else matters

- `NQ` and `ND` are declared small (`public.scales`); `NBF 56` is the order-6 basis size.

## Before calling it done

- [x] Oracle, manifest, DaCe program, `initialize()`, this record, README.
- [x] Route replayed and built at S on CPU and GPU, oracle checked.
- [ ] Timed at M (1 thread, 72 threads, GPU): baseline, reference, moves 1 to 3, a swap decoy.
