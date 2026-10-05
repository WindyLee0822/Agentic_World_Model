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


def test_b2_samples_real_env_and_selects_by_real_success():
    """B2 interacts with the REAL env during sampling and selects best-of-N by the **real-env**
    success flag (the grounded reward), committing a real-successful rollout when one exists."""
    # _FairFakeEnv.run_agent_rollout: a rollout succeeds iff its seed is even.
    calls: List[int] = []
    env = _FairFakeEnv(calls)
    cfg = Config(method="b2", env="fake", N=4, seed=0, solver=SolverConfig(agent="llm"))
    out = run_decision(env, cfg)
    r = out.record
    assert sorted(calls) == [0, 1, 2, 3]                 # sampled N real-env rollouts
    assert out.rollouts[out.selected_index].grounded.success is True  # selected a real success
    assert r.success is True and r.reward == 1.0          # committed outcome = real-env success
    assert r.simulator_writes == 0 and r.n_commits == 1

    # when NO rollout succeeds (all odd seeds), B2 honestly reports failure (no false success)
    env2 = _FairFakeEnv([])
    cfg2 = Config(method="b2", env="fake", N=2, seed=1, solver=SolverConfig(agent="llm"))  # seeds 1,2
    out2 = run_decision(env2, cfg2)
    # seeds 1 and 2 -> one success (2); B2 must pick it
    assert out2.record.success is True
    env3 = _FairFakeEnv([])
    cfg3 = Config(method="b2", env="fake", N=1, seed=1, solver=SolverConfig(agent="llm"))  # seed 1 only
    assert run_decision(env3, cfg3).record.success is False  # only a failing rollout available


def test_opt_in_self_aggregation_bypasses_selector_and_keeps_legacy_default(monkeypatch):
    import tts.harness as harness
    from tts.aggregation import AggregationResult

    original_select = harness.Selector.select
    legacy = run_decision(
        _FairFakeEnv([]),
        Config(method="b2", env="fake", N=2, solver=SolverConfig(agent="llm")),
    )
    assert legacy.selected_index in (0, 1)

    monkeypatch.setattr(
        harness.Selector,
        "select",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selector must not run in self-aggregation mode")
        ),
    )
    monkeypatch.setattr(
        harness,
        "aggregate_trajectories",
        lambda env, config, rollouts: AggregationResult(
            rollouts[1].trajectory, source_index=1, tokens=23
        ),
    )
    aggregated = run_decision(
        _FairFakeEnv([]),
        Config(
            method="b2",
            env="fake",
            N=2,
            solver=SolverConfig(agent="llm", self_aggregate=True),
        ),
    )
    assert aggregated.selected_index == 1
    assert aggregated.record.n_rollouts == 2
    assert aggregated.record.n_commits == 1
    assert aggregated.record.tokens == 23

    monkeypatch.setattr(harness.Selector, "select", original_select)


def test_tau2_wm_agent_policy_hook_is_disabled():
    """τ² WM sampling must not inject verifier guidance into the official agent prompt."""
    from tts.env.real_tau2 import _add_wm_agent_policy_reminder

    class Agent:
        def __init__(self):
            self.get_init_state = object()

    agent = Agent()
    original = agent.get_init_state
    _add_wm_agent_policy_reminder(agent, "airline")
    assert agent.get_init_state is original


def test_simulator_prompt_changes_are_isolated_from_agent_prompts():
    """Simulator discipline is domain-specific and never appended to benchmark-agent prompts."""
    from tts.env.appworld_wm import _APPWORLD_SIMULATOR_SYSTEM
    from tts.env.officebench_agent import PROMPT_DICT
    from tts.env.officebench_wm_predict import _OFFICEBENCH_SIMULATOR_SYSTEM
    from tts.env.real_tau2 import _TAU2_SIMULATOR_SYSTEM

    simulator_text = "\n".join((
        _TAU2_SIMULATOR_SYSTEM,
        _APPWORLD_SIMULATOR_SYSTEM,
        _OFFICEBENCH_SIMULATOR_SYSTEM,
    ))
    assert "not an assistant" in simulator_text
    assert "coach the benchmark agent" in simulator_text
    assert all(
        "causal simulator" not in str(prompt).lower()
        for prompt in PROMPT_DICT.values()
    )


def test_tau2_real_and_wm_use_same_agent_sampling_config():
    """Both paths share one run config; Qwen customization is method-independent."""
    import inspect

    from tts.env.real_tau2 import Tau2TaskEnv

    real_source = inspect.getsource(Tau2TaskEnv.run_agent_rollout)
    wm_source = inspect.getsource(Tau2TaskEnv.run_agent_rollout_wm)

    assert "self._build_run_cfg(config, seed, default_model)" in real_source
    assert "self._build_run_cfg(config, seed, default_model)" in wm_source
    assert "_add_wm_agent_policy_reminder(" not in wm_source
    assert "_add_airline_cancel_preflight(" not in wm_source
    assert "_add_airline_baggage_preflight(" not in wm_source
    assert "_add_airline_certificate_preflight(" not in wm_source
    assert "orch.environment.get_response =" in wm_source


def test_tau2_wm_reference_audit_rejects_duplicate_writes():
    """Selection quality comes from the WM verifier, not from steering the τ² agent."""
    from tts.env.real_tau2 import Tau2ShadowEvaluator

    required = Action("cancel_reservation", {"reservation_id": "ABC123"})
    evaluator = Tau2ShadowEvaluator(
        "airline",
        [required],
        model=object(),
        gold_action_specs=[required],
    )
    exact = Trajectory(steps=[Step(required)])
    duplicate = Trajectory(steps=[Step(required), Step(required)])

    exact_mismatch, exact_score = evaluator._reference_assessment(exact)
    duplicate_mismatch, duplicate_score = evaluator._reference_assessment(duplicate)

    assert exact_mismatch is None and exact_score == 1.0
    assert "duplicate or unreferenced" in duplicate_mismatch
    assert duplicate_score < exact_score
