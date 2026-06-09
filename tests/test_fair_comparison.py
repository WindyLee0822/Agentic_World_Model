"""B1/B2/Ours must use the SAME agent prompting strategy for a fair comparison.

The real-agent rollout path is method-independent (it depends only on ``config.solver``, never on
``config.method``), and with a shared ``rollout_cache`` every method selects from the **same pool**
of agent trajectories.  These tests pin that contract down with a pure in-memory fake env (no
benchmark packages / no LLM), so the fairness guarantee is enforced even offline.
"""

from __future__ import annotations

from typing import List

from tts.abcs import RestrictedView, TaskEnvironment
from tts.config import Config, SolverConfig
from tts.harness import run_decision
from tts.trajectory import Observation, RewardInfo, Step, Trajectory
from tts.types import Action


class _View(RestrictedView):
    @property
    def task_id(self) -> str:
        return "t0"

    def instruction(self) -> str:
        return "do the task"

    def tool_schema(self) -> List[str]:
        return ["act"]


class _FairFakeEnv(TaskEnvironment):
    """A fake env whose 'real agent' returns a deterministic, seed-dependent trajectory.

    ``calls`` is a shared list recording every seed actually generated, so a test can assert the
    agent was invoked once per distinct seed (i.e. the pool is shared across methods)."""

    subset = "fake"

    def __init__(self, calls: List[int]) -> None:
        self._calls = calls

    @property
    def env_name(self) -> str:
        return "fake_fair"

    @property
    def task_id(self) -> str:
        return "t0"

    def restricted_view(self) -> RestrictedView:
        return _View()

    def entity_store(self):  # pragma: no cover - not used (world model unsupported)
        raise NotImplementedError

    def world_model(self, config, budget, cache=None):
        raise NotImplementedError  # forces B1/Ours to the oracle fallback (same pool)

    def fork(self):  # pragma: no cover
        raise NotImplementedError

    def gold_trajectory(self) -> Trajectory:
        return Trajectory(steps=[])

    def run_agent_rollout(self, config, seed: int):
        self._calls.append(seed)  # records that the agent was actually prompted for this seed
        traj = Trajectory(steps=[Step(Action("act", {"seed": seed}))], final_message="done")
        grounded = RewardInfo(reward=1.0 if seed % 2 == 0 else 0.0, success=(seed % 2 == 0),
                              grounded=True, reason="fake official reward")
        return traj, grounded

    def step(self, action: Action) -> Observation:  # pragma: no cover
        return Observation(content="", error=False)

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:  # pragma: no cover
        return RewardInfo(0.0, False, grounded=True)

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:  # pragma: no cover
        return RewardInfo(0.0, False, grounded=True)

    def state_hash(self) -> str:
        return "h"

    def committed_count(self) -> int:
        return 0


def test_b1_b2_ours_share_one_agent_rollout_pool():
    """All three methods select from the identical pool of agent rollouts (same prompt + seeds)."""
    calls: List[int] = []
    cache: dict = {}
    pools = {}
    for method in ("b1", "b2", "ours"):
        env = _FairFakeEnv(calls)  # fresh instance per method, like experiments/run.py
        cfg = Config(method=method, env="fake", N=4, seed=0,
                     solver=SolverConfig(agent="llm"))
        out = run_decision(env, cfg, rollout_cache=cache)
        pools[method] = [r.trajectory.signature() for r in out.rollouts]
        assert out.record.n_commits == 1 and out.record.simulator_writes == 0

    # the agent was prompted exactly once per distinct seed (0..3) — the pool is shared, not re-sampled
    assert sorted(calls) == [0, 1, 2, 3]
    # and every method saw the identical agent trajectories (identical prompting strategy)
    assert pools["b1"] == pools["b2"] == pools["ours"]


def test_agent_rollout_generation_is_method_independent():
    """Without a shared cache, each method still uses the same method-independent prompting path."""
    sigs = {}
    for method in ("b0", "b1", "b2", "ours"):
        calls: List[int] = []
        env = _FairFakeEnv(calls)
        n = 1 if method == "b0" else 3
        cfg = Config(method=method, env="fake", N=n, seed=0, solver=SolverConfig(agent="llm"))
        out = run_decision(env, cfg)
        sigs[method] = [r.trajectory.signature() for r in out.rollouts]
        # generation depends only on the seed, never on the method
        assert sorted(calls) == list(range(n))
    # the first rollout (seed 0) is identical for every method
    assert {sigs[m][0] for m in sigs} == {sigs["b0"][0]}
