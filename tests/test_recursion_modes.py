"""Recursion stop rules: uncertainty (conf-driven), fixed-depth = L, and the hard cap."""

from __future__ import annotations

import pytest

from tts.trajectory import Trajectory
from conftest import make_chain_wm, make_feasible_infeasible_env, make_recursive_wm


def test_uncertainty_spawns_on_low_conf_not_on_high():
    """Root over a multi-leaf item has low read-coverage -> spawns; leaf nodes are complete -> don't."""
    env = make_feasible_infeasible_env()
    wm = make_recursive_wm(env, mode="uncertainty", tau=0.9, node_read_limit=1)
    wm.evaluate(env.gold_trajectory())  # gold selects item 'A' with 3 leaves
    root = wm.spawn_root
    assert root.spawned, "root has unread leaves (low coverage) -> must spawn"
    leaves = [n for n in root.walk() if n.region.startswith("d") and not n.children]
    assert leaves, "expected leaf probe nodes"
    assert all(not n.spawned for n in leaves), "fully-covered leaf nodes must not spawn"


def test_uncertainty_high_tau_reads_more_than_low_tau():
    env = make_feasible_infeasible_env()
    lo = make_recursive_wm(env, mode="uncertainty", tau=0.1, node_read_limit=1)
    lo.evaluate(env.gold_trajectory())
    hi = make_recursive_wm(env, mode="uncertainty", tau=0.99, node_read_limit=1)
    hi.evaluate(env.gold_trajectory())
    assert hi.spawn_root.total_reads() >= lo.spawn_root.total_reads()


@pytest.mark.parametrize("L", [0, 1, 2, 3, 4])
def test_fixed_mode_descends_exactly_L_layers(L):
    wm = make_chain_wm(mode="fixed", L=L, hard_cap=99, branching=1, node_read_limit=1)
    wm.evaluate(Trajectory(steps=[]))
    assert wm.spawn_root.max_depth() == L


def test_hard_cap_bounds_depth():
    wm = make_chain_wm(mode="fixed", L=50, hard_cap=3, branching=1, node_read_limit=1)
    wm.evaluate(Trajectory(steps=[]))
    assert wm.spawn_root.max_depth() <= 3
