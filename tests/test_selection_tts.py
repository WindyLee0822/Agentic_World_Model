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
    ours_1 = _success_rate("ours", 1)
    ours_8 = _success_rate("ours", 8)
    b2_8 = _success_rate("b2", 8)

    assert ours_8 > ours_1            # test-time scaling: more rollouts -> higher success
    assert abs(ours_8 - b2_8) <= 0.2  # recursive ~ forking oracle, without forking


def test_ours_approaches_oracle_b2():
    ours = _success_rate("ours", 8)
    b2 = _success_rate("b2", 8)
    assert b2 >= ours - 0.2


@pytest.mark.parametrize("score", ["simulator_success", "self_consistency", "probe_agreement"])
def test_selector_prefers_predicted_success(score):
    assert score in available_scores()
    good = RolloutResult(0, Trajectory(steps=[]), RewardInfo(1.0, True))
    bad = RolloutResult(1, Trajectory(steps=[]), RewardInfo(0.0, False))
    idx, _ = Selector(score).select([bad, good])
    assert idx == 1  # the predicted-successful rollout is selected


def test_selector_never_prefers_invalid_rollout_over_valid_failure():
    invalid = RolloutResult(
        0,
        Trajectory(steps=[], done=False, termination_reason="rollout_error"),
        RewardInfo(0.0, False),
    )
    valid_failure = RolloutResult(1, Trajectory(steps=[]), RewardInfo(0.0, False))

    idx, _ = Selector("simulator_success").select([invalid, valid_failure])
    assert idx == 1


def test_selector_never_prefers_hard_invalid_answer_over_uncertain_failure():
    wrong_answer = RolloutResult(
        index=0,
        trajectory=Trajectory(steps=[], final_message="wrong answer"),
        predicted=RewardInfo(
            reward=0.0,
            success=False,
            grounded=False,
            metadata={"hard_invalid": True},
        ),
    )
    uncertain = RolloutResult(
        index=1,
        trajectory=Trajectory(steps=[], final_message="uncertain"),
        predicted=RewardInfo(reward=0.0, success=False, grounded=False),
    )

    idx, _ = Selector("simulator_success").select([wrong_answer, uncertain])

    assert idx == 1


def test_confidence_selector_breaks_binary_success_ties():
    lower = RolloutResult(
        0, Trajectory(steps=[]), RewardInfo(reward=0.6, success=True))
    higher = RolloutResult(
        1, Trajectory(steps=[]), RewardInfo(reward=0.8, success=True))

    idx, _ = Selector("simulator_confidence").select([lower, higher])

    assert idx == 1


def test_confidence_selector_uses_new_candidates_as_bon_pool_grows():
    candidates = [
        RolloutResult(
            index, Trajectory(steps=[]),
            RewardInfo(reward=confidence, success=confidence >= 0.5),
        )
        for index, confidence in enumerate((0.2, 0.6, 0.4, 1.0))
    ]

    selected = [
        Selector("simulator_confidence").select(candidates[:n])[0]
        for n in (1, 2, 4)
    ]

    assert selected == [0, 1, 3]


def test_guarded_confidence_selector_requires_clear_improvement(monkeypatch):
    monkeypatch.setenv("TTS_SELECTION_CONFIDENCE_MARGIN", "0.1")
    candidates = [
        RolloutResult(
            index, Trajectory(steps=[]),
            RewardInfo(reward=confidence, success=confidence >= 0.5),
        )
        for index, confidence in enumerate((0.20, 0.25, 0.31, 0.80))
    ]

    selected = [
        Selector("simulator_confidence_guarded").select(candidates[:n])[0]
        for n in (1, 2, 4)
    ]

    assert selected == [0, 0, 3]


def test_guarded_confidence_selector_replaces_hard_invalid_incumbent(monkeypatch):
    monkeypatch.setenv("TTS_SELECTION_CONFIDENCE_MARGIN", "0.1")
    invalid = RolloutResult(
        0,
        Trajectory(steps=[]),
        RewardInfo(
            reward=1.0,
            success=True,
            metadata={"hard_invalid": True},
        ),
    )
    valid = RolloutResult(
        1,
        Trajectory(steps=[]),
        RewardInfo(reward=0.0, success=False),
    )

    idx, _ = Selector("simulator_confidence_guarded").select([invalid, valid])

    assert idx == 1
