# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NestForge user plugin, registered with the bench once per test process: the bench registers a plugin's
packets once, so two modules that each cleared its caches and loaded it again would register them twice."""

from collections.abc import Iterator

import pytest
from hpcagent_agent import plugins as agent_plugins
from hpcagent_agent.plugins import Plugin
from hpcagent_bench import plugins
from hpcagent_bench import study_tags as tags

from nestforge.paths import REPO_ROOT

#: The plugin folder HPCAgent-Bench loads.
PLUGIN = REPO_ROOT / "plugin" / "nestforge"


@pytest.fixture(scope="session")
def nestforge_plugin() -> Iterator[tuple[Plugin, ...]]:
    """Every plugin the bench found with NestForge's folder as the user plugin dir, packets, routes and tools loaded."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(PLUGIN))
        plugins.register.cache_clear()
        plugins.load.cache_clear()
        tags.registry.cache_clear()
        found = plugins.load()
        agent_plugins.import_tools()
        yield found
