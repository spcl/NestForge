# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A reference solution's stage 4: ``Script.place``, ``Script.insert_copies`` and ``Script.unplace`` drive the session's
placement, record each call, and raise when the session refuses one. Transformations only: no kernel is built."""

import pytest
from test_unplace import TARGETS, doalls, fresh

from nestforge.replay import ReplayError, Script


@pytest.mark.parametrize("gpu", TARGETS)
def test_place_records_every_kernels_device_and_scopes_what_no_scope_covers(tmp_path, gpu):
    script = Script(fresh(gpu, tmp_path))
    first, _ = doalls(script.session)
    kernel = script.session.set_scope([first]).reason

    done = script.place({kernel: "cpu"})

    assert done.kind == "place" and f"{kernel}=cpu" in done.reason
    assert len(script.session.devices()) == 2 and script.applied == [done]


def test_insert_copies_then_unplace_go_back_to_the_kernels_before_placement(tmp_path):
    script = Script(fresh(True, tmp_path))
    script.place()
    kernels = sorted(script.session.devices())

    script.insert_copies()
    script.unplace()

    assert [move.kind for move in script.applied] == ["place", "insert-copies", "unplace"]
    assert script.session.unplaced is None and sorted(script.session.devices()) == kernels


@pytest.mark.parametrize("gpu", TARGETS)
def test_a_refused_placement_call_raises_and_records_nothing(tmp_path, gpu):
    script = Script(fresh(gpu, tmp_path))

    with pytest.raises(ReplayError, match="unplace: illegal: no kernel is placed"):
        script.unplace()
    with pytest.raises(ReplayError, match="insert-copies: illegal"):
        script.insert_copies()
    assert script.applied == []


def test_a_device_the_target_lacks_is_refused(tmp_path):
    script = Script(fresh(False, tmp_path))
    first, _ = doalls(script.session)
    kernel = script.session.set_scope([first]).reason

    with pytest.raises(ReplayError, match="place: illegal"):
        script.place({kernel: "gpu"})
