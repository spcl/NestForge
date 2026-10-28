# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""SeisSol volume update, written output-stationary (``programs/seissol_volume_dace.py``): on the GPU, split the
init from the sum; on the CPU, also push the sum map inside the summed loops ``l`` and ``d``."""

from nestforge.replay import Context, Script

KERNEL = "seissol_tensor_contraction"
PROGRAM = "programs/seissol_volume_dace.py"
#: The program's basis, quantity and dimension symbols at the corpus kernel's fixed values (order 7).
PARAMS = {"NBF": 84, "NQ": 9, "ND": 3}
CONFIGS = (Context("cpu", 72, "M"), Context("gpu", 1, "M"))
CI_SKIP = (
    "not replayed yet: at preset S on one CPU thread the replay does not finish within 25 minutes on DaCe 7683e143"
)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("map-fission", writes=["upd"]))
    if ctx.device == "cpu":
        s.apply(s.pick("interchange-map-loop", writes=["upd"]))
        s.apply(s.pick("interchange-map-loop", writes=["upd"]))
    s.finish()
