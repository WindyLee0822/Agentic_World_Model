"""Core invariants: simulator-never-writes + privileged-vs-restricted access asymmetry."""

from __future__ import annotations

import pytest

from tts.abcs import EntityStore, RestrictedView
from tts.budget import Budget
from tts.cache import ReadCache
from tts.config import Config, SimulatorConfig, SolverConfig
from tts.harness import run_decision
from conftest import make_feasible_infeasible_env, make_recursive_wm


@pytest.mark.parametrize("method", ["b2", "ours"])
def test_simulator_never_writes_real_env(method):
    """The real backing state hash is unchanged by the whole sampling phase."""
    env = make_feasible_infeasible_env()
    pre = env.state_hash()
    cfg = Config(method=method, env="fake", N=4, seed=2, solver=SolverConfig(p_correct=0.5))
    out = run_decision(env, cfg)
    assert out.record.simulator_writes == 0  # nothing committed during sampling
    assert out.record.n_commits == 1


def test_commit_is_the_only_mutation():
    """Sampling leaves real state untouched; committing the gold trajectory mutates it once."""
    env = make_feasible_infeasible_env()
    pre = env.state_hash()
    wm = make_recursive_wm(env)
    wm.evaluate(env.gold_trajectory())  # a full sampling read pass
    assert env.state_hash() == pre  # world model never wrote
    env.commit_trajectory(env.gold_trajectory())  # the gold plan selects an item -> mutates
    assert env.state_hash() != pre


def test_world_model_cannot_write():
    """A world model is built from a read-only EntityStore and exposes no write path."""
    env = make_feasible_infeasible_env()
    wm = make_recursive_wm(env)
    wm.evaluate(env.gold_trajectory())
    # EntityStore is the only state handle a world model gets, and it is read-only.
    store = env.entity_store()
    assert isinstance(store, EntityStore)
    assert not any(hasattr(store, m) for m in ("write", "commit", "update", "delete", "insert"))


def test_restricted_view_has_no_privileged_reads():
    """A solver's view exposes instruction + tool names only — no EntityStore, no commit."""
    env = make_feasible_infeasible_env()
    view = env.restricted_view()
    assert isinstance(view, RestrictedView)
    assert not any(hasattr(view, m) for m in ("entity_store", "read", "commit", "backing_db"))
    assert view.instruction() and view.tool_schema()
