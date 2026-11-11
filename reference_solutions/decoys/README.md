# Decoys (branch `decoys` only)

One file per kernel, named like its reference solution in the folder above. Each exports `DECOYS: dict[str, Decoy]`: schedules an agent could reasonably take that are not the reference, with the mistake it is (`Mistake` in `nestforge/replay.py`) and why it is tempting. Replay times them next to the baseline and the reference, so their cost is measured, not written here.

This folder and the `Decoy` code in `replay.py` exist on the long-lived `decoys` branch, not on `main`: agent runs are traced, so main declares no wrong turns. Replay on `main` ignores this folder.

Keeping the branch current: after each merge to `main`, `git rebase origin/main` on `decoys`. Only `nestforge/replay.py`, `tests/test_replay.py` and the files in this folder differ from `main`, so a conflict can only be in the first two.
