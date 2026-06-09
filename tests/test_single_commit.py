"""Single-commit invariant: each decision commits exactly one trajectory, at most once."""

from __future__ import annotations

import pytest

from tts.commit import CommitChokepoint
from tts.config import Config, SolverConfig
from tts.harness import run_decision
from conftest import make_feasible_infeasible_env


@pytest.mark.parametrize("method", ["b0", "b1", "b2", "ours"])
def test_exactly_one_commit(method):
    env = make_feasible_infeasible_env()
    cfg = Config(method=method, env="fake", N=4, seed=1, solver=SolverConfig(p_correct=0.5))
    out = run_decision(env, cfg)
    assert out.record.n_commits == 1
    assert env.committed_count() == 1


def test_chokepoint_rejects_second_commit():
    env = make_feasible_infeasible_env()
    chokepoint = CommitChokepoint(env)
    traj = env.gold_trajectory()
    chokepoint.commit(traj)
    with pytest.raises(RuntimeError):
        chokepoint.commit(traj)
