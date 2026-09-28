# Stage 1: Canonicalize

[Overview](../../README.md) · next: [2 Moves](2-moves.md)

DaCe canonicalization rewrites the program into the canonical parallel form (CPF): parallel loops become maps,
reductions and scans are lifted, statements are distributed. It runs once, with the target's preset, up to and
excluding the `fuse` stage, so stage 2 owns every fusion decision. It is never searched.

| | |
|---|---|
| default | `Session.canonicalize()` |
| agent | none |
| code | `nestforge/stages/canonicalize.py` |
