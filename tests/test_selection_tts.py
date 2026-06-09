"""TTS separation (B0 <= B1 < Ours ~= B2) + selection rankers, on the in-memory fake env."""

from __future__ import annotations

import statistics

import pytest

from tts.config import Config, SimulatorConfig, SolverConfig
from tts.harness import run_decision
from tts.selection import Selector, available_scores
from tts.trajectory import RewardInfo, RolloutResult, Trajectory
from tts.types import Action
from conftest import make_feasible_infeasible_env


def _success_rate(method, N, ntasks=12, p=0.5):
    succ = []
    for t in range(ntasks):
        env = make_feasible_infeasible_env(task_id=f"fake-{t}")
        cfg = Config(method=method, env="fake", N=N, seed=t * 1000,
                     success_score="simulator_success", solver=SolverConfig(p_correct=p),
                     simulator=SimulatorConfig(recursion_mode="uncertainty", tau=0.9, node_read_limit=1))
        succ.append(run_decision(env, cfg).record.success)
    return statistics.fmean(succ)


def test_tts_curve_separation():
    """Ours and B2 rise with N; B0 and B1 stay near the base rate (flat under-reads)."""
    b0 = _success_rate("b0", 8)
    b1_8 = _success_rate("b1", 8)
    ours_1 = _success_rate("ours", 1)
    ours_8 = _success_rate("ours", 8)
    b2_8 = _success_rate("b2", 8)

    assert ours_8 > ours_1            # test-time scaling: more rollouts -> higher success
    assert ours_8 >= b1_8            # recursive beats flat
    assert ours_8 > b0               # recursive beats the single-rollout baseline
    assert abs(ours_8 - b2_8) <= 0.2  # recursive ~ forking oracle, without forking


def test_ours_matches_oracle_b2_better_than_flat_b1():
    b1 = _success_rate("b1", 8)
    ours = _success_rate("ours", 8)
    b2 = _success_rate("b2", 8)
    assert b1 <= ours
    assert b2 >= ours - 0.2


@pytest.mark.parametrize("score", ["simulator_success", "self_consistency", "probe_agreement"])
def test_selector_prefers_predicted_success(score):
    assert score in available_scores()
    good = RolloutResult(0, Trajectory(steps=[]), RewardInfo(1.0, True))
    bad = RolloutResult(1, Trajectory(steps=[]), RewardInfo(0.0, False))
    idx, _ = Selector(score).select([bad, good])
    assert idx == 1  # the predicted-successful rollout is selected
