"""Read cache (hit/miss correctness) + budget kill-switch."""

from __future__ import annotations

from tts.budget import Budget
from tts.cache import ReadCache
from tts.config import BudgetConfig, SimulatorConfig
from tts.worldmodel import make_world_model
from conftest import FakeEntityStore, FakeShadowEvaluator, make_feasible_infeasible_env
from tts.types import Action


def _eval_with(env, cache, budget, kind="recursive"):
    cfg = SimulatorConfig(kind=kind, recursion_mode="uncertainty", tau=0.9, node_read_limit=1, hard_cap=6)
    wm = env.world_model(cfg, budget, cache)
    return wm.evaluate(env.gold_trajectory())


def test_cache_hit_avoids_backing_reads():
    env = make_feasible_infeasible_env()
    cache = ReadCache(enabled=True)
    budget = Budget()
    _eval_with(env, cache, budget)
    reads_after_first = budget.n_reads
    assert reads_after_first > 0 and cache.misses > 0
    # second evaluation of the same trajectory: every entity is now cached -> no new backing reads
    _eval_with(env, cache, budget)
    assert budget.n_reads == reads_after_first  # no additional backing reads
    assert cache.hits > 0


def test_cache_disabled_re_reads():
    env = make_feasible_infeasible_env()
    budget = Budget()
    _eval_with(env, ReadCache(enabled=False), budget)
    first = budget.n_reads
    _eval_with(env, ReadCache(enabled=False), budget)
    assert budget.n_reads > first  # re-reads with no cache


def test_flat_reads_less_than_recursive():
    env = make_feasible_infeasible_env()
    b_flat, b_rec = Budget(), Budget()
    cfg_flat = SimulatorConfig(kind="flat", node_read_limit=1)
    cfg_rec = SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9, node_read_limit=1, hard_cap=6)
    env.world_model(cfg_flat, b_flat, ReadCache()).evaluate(env.gold_trajectory())
    env.world_model(cfg_rec, b_rec, ReadCache()).evaluate(env.gold_trajectory())
    assert b_flat.n_reads < b_rec.n_reads


def test_budget_kill_switch_caps_spawns():
    env = make_feasible_infeasible_env()
    budget = Budget(max_spawns=1)
    cfg = SimulatorConfig(kind="recursive", recursion_mode="fixed", L=5, node_read_limit=1, hard_cap=99, branching=2)
    wm = env.world_model(cfg, budget, ReadCache())
    wm.evaluate(env.gold_trajectory())
    assert budget.kill_switch_trips >= 1
    assert budget.n_spawns <= 1
