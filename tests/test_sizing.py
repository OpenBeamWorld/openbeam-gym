# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Every published robot, read back from its zip as the game reads it,
against the game's fixed-step budgets (openbeam_gym/robots/sizing.py)."""

import json
import zipfile
from pathlib import Path

import pytest

from openbeam_gym.robots import build_camera_boom, build_humanoid, build_test_arm, sizing

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def published_part(name):
    with zipfile.ZipFile(ASSETS / f"{name}.zip") as z:
        return json.loads(z.read(f"vehicles/{name}/{name}.jbeam"))[name]


@pytest.mark.parametrize("module", [build_camera_boom, build_humanoid])
def test_sized_robots_stay_within_the_budgets(module):
    report = sizing.budget_report(published_part(module.NAME))
    assert report["within_budget"]
    assert report["torsion_bars_scaled"] == 0 and report["smallest_torsion_scale"] == 1.0
    for node, load in report["loads"].items():
        assert load["torsion"] <= 0.7, node
        assert load["beams"] <= 2.0 + 1e-9, node
    # The builder's own bookkeeping agrees with the package.
    own = module.node_loads(module.build())
    for node, load in own.items():
        assert load["torsion"] == pytest.approx(report["loads"][node]["torsion"], rel=1e-3, abs=1e-9)


def test_the_law_reproduces_the_games_measurement_of_the_earlier_boom():
    # The earlier boom (every node 3 kg, 300 kN m/rad and 800 N m s/rad per
    # bar) logged worstAuthoredNodeCriterion=63.2405, smallestScale=0.0127.
    boom = build_camera_boom.build()
    masses = {n: 3.0 for n in boom.nodes}
    loads = sizing.node_loads(boom.nodes, masses, {}, [(h["nodes"], 300000.0, 800.0) for h in boom.hydros])
    worst = max(v["torsion"] for v in loads.values())
    assert worst == pytest.approx(63.2405, rel=2e-3)
    assert sizing.TORSION_GAME_LIMIT / worst == pytest.approx(0.0127, abs=5e-5)


def test_the_test_arm_is_reported_and_left_as_it_is():
    # The test arm keeps its physics (its PPO results stay comparable); it asks
    # for more than the budgets, and the game scales it at spawn.
    report = build_test_arm.budget()
    assert not report["within_budget"]
    assert report["worst_torsion"][1] == pytest.approx(14.0625)
    assert report["torsion_bars_scaled"] == 16
    assert report["smallest_torsion_scale"] == pytest.approx(0.8 / 14.0625)
    assert report["worst_beams"][1] == pytest.approx(5.0) and report["beam_nodes_over_limit"] == 8
    assert sizing.budget_report(published_part(build_test_arm.NAME))["loads"] == report["loads"]


def test_progressive_fill_respects_the_ceiling():
    factors = {0: {"a": 2.0, "b": 1.0}, 1: {"b": 1.0, "c": 4.0}}
    q = sizing.progressive_fill({0: 1.0, 1: 1.0}, factors, 0.7)
    load = {}
    for b, f in factors.items():
        for n, v in f.items():
            load[n] = load.get(n, 0.0) + v * q[b]
    assert max(load.values()) == pytest.approx(0.7)
    assert all(v <= 0.7 + 1e-12 for v in load.values())
