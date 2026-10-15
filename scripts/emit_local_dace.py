# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate ``<name>_dace.py`` of a repo-local benchmark from its ``<name>_numpy.py``, as hpcagent_bench does for its
corpus, but next to the kernel and not under hpcagent_bench's ``benchmarks/``.

Usage: ``python scripts/emit_local_dace.py <name>``. Review the output and commit it; NestForge never
regenerates it when it loads a kernel. hpcagent_bench is imported inside functions, like the rest of NestForge."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import yaml

from nestforge.corpus.bench import LOCAL_ROOT


def emit(name: str) -> Path:
    """Write ``<name>_dace.py`` next to ``<name>_numpy.py`` and return its path."""
    from hpcagent_bench import emit_bridge
    from hpcagent_bench.spec import BenchSpec
    from numpyto_c.dace_emit import emit_dace
    from numpyto_common.emit_io import write_generated
    from numpyto_common.frontend import emit_with_inline_fallback, parse_kernel

    folder = LOCAL_ROOT / name
    yaml_path = folder / f"{name}.yaml"
    numpy_path = folder / f"{name}_numpy.py"
    manifest: dict[str, Any] = yaml.safe_load(yaml_path.read_text())
    spec = BenchSpec.from_yaml(manifest, source=str(yaml_path))
    out = folder / f"{name}_dace.py"

    def render() -> str:
        with emit_bridge.bench_info_tempfile(spec) as bench_info:
            rendered = emit_dace(parse_kernel(numpy_path, bench_info, open_mesh_grids=False))
        ast.parse(rendered)  # syntactic self-check before writing
        return rendered

    write_generated(out, emit_with_inline_fallback(render), source=numpy_path.name)
    return out


def main() -> None:
    print(emit(sys.argv[1]))


if __name__ == "__main__":
    main()
