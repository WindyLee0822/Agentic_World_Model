"""Guarded tests against the REAL benchmarks (skipped when a benchmark isn't installed).

τ²-bench runs in the ``dw`` interpreter; AppWorld in the ``env`` interpreter (with
``APPWORLD_ROOT`` set).  These assert the trajectory pipeline coheres to the official execution
and that the read-grounded TTS separation holds on real state.
"""

from __future__ import annotations

import os

import pytest

from tts.config import Config, SimulatorConfig, SolverConfig
from tts.harness import run_decision

try:
    from tts.env.real_tau2 import real_tau2_available
except Exception:  # pragma: no cover
    real_tau2_available = lambda: False  # noqa: E731

try:
    from tts.env.real_appworld import real_appworld_available
except Exception:  # pragma: no cover
    real_appworld_available = lambda: False  # noqa: E731

tau2 = pytest.mark.skipif(not real_tau2_available(), reason="real tau2 not available")
appworld = pytest.mark.skipif(not real_appworld_available(), reason="real appworld not available")
needs_llm = pytest.mark.skipif(
    not (os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_API_KEY")),
    reason="no LLM endpoint configured (set AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY)",
)


@tau2
def test_tau2_official_gold_trajectory_scores_one():
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    gold = env.gold_trajectory()
    ri = env.evaluate(gold)
    assert ri.grounded and ri.success and ri.reward == 1.0  # official ENV evaluation


@tau2
def test_tau2_world_model_never_writes_real_db():
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    pre = env.state_hash()
    cfg = Config(method="ours", env="tau2", N=4, seed=0, solver=SolverConfig(p_correct=0.5),
                 simulator=SimulatorConfig(recursion_mode="uncertainty", tau=0.9, node_read_limit=1))
    out = run_decision(env, cfg)
    assert out.record.simulator_writes == 0
    assert out.record.n_commits == 1


@tau2
def test_tau2_recursive_predicts_better_than_flat():
    """On multi-entity tasks the flat world model under-reads; the recursive one is accurate."""
    from tts.env.registry import make_env

    flat_ok = rec_ok = 0
    for idx in range(6):
        env_f = make_env("tau2", index=idx)
        env_r = make_env("tau2", index=idx)
        gold = env_f.gold_trajectory()
        bcfg_flat = SimulatorConfig(kind="flat", node_read_limit=1)
        bcfg_rec = SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9, node_read_limit=1, hard_cap=8)
        from tts.budget import Budget
        from tts.cache import ReadCache

        flat = env_f.world_model(bcfg_flat, Budget(), ReadCache()).evaluate(gold)
        rec = env_r.world_model(bcfg_rec, Budget(), ReadCache()).evaluate(gold)
        flat_ok += int(flat.success)
        rec_ok += int(rec.success)
    assert rec_ok >= flat_ok and rec_ok >= 4  # recursive certifies the gold trajectories


@tau2
@needs_llm
def test_tau2_b0_real_official_agent_runs():
    """B0 drives the official τ² agent loop (official prompt) and returns a grounded reward."""
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    cfg = Config(method="b0", env="tau2", N=1, seed=0,
                 solver=SolverConfig(agent="llm", max_steps=30))
    out = run_decision(env, cfg)
    r = out.record
    assert r.n_commits == 1 and r.simulator_writes == 0
    assert r.reward in (0.0, 1.0)  # official EvaluationType.ALL reward


@appworld
def test_appworld_real_task_metadata_loads():
    """The AppWorld env exposes the REAL task instruction (no analog playlist task)."""
    from tts.env.real_appworld import AppWorldTaskEnv

    env = AppWorldTaskEnv(split="dev", level=1, index=0)
    assert env.task_id and isinstance(env.restricted_view().instruction(), str)
    assert env.restricted_view().tool_schema() == ["python"]
    env.close()


@appworld
def test_appworld_outputs_are_cleaned_up():
    """Opening an AppWorld task must not leave its on-disk output dir behind (no temp-file growth)."""
    import os

    from tts.env.real_appworld import AppWorldTaskEnv, _experiment_outputs_root

    root = _experiment_outputs_root()
    base = "tts_cleanup_probe"
    before = {d for d in os.listdir(root)} if root and os.path.isdir(root) else set()
    env = AppWorldTaskEnv(split="dev", level=1, index=0, experiment_name=base)
    env.close()
    after = {d for d in os.listdir(root)} if root and os.path.isdir(root) else set()
    # the metadata-read world's output dir was removed on exit
    assert not any(d.startswith(base) for d in after - before)


def test_appworld_sweep_removes_only_matching_base(tmp_path, monkeypatch):
    """sweep_experiment_outputs removes dirs with the given base prefix and nothing else."""
    import os

    from tts.env import real_appworld

    root = tmp_path / "experiments" / "outputs"
    root.mkdir(parents=True)
    monkeypatch.setattr(real_appworld, "_experiment_outputs_root", lambda: str(root))
    for name in ("run_A_x", "run_A_y", "run_B_z"):
        (root / name).mkdir()
    removed = real_appworld.sweep_experiment_outputs("run_A")
    assert removed == 2
    assert {p.name for p in root.iterdir()} == {"run_B_z"}
    # honour the keep-outputs opt-out
    monkeypatch.setenv("TTS_KEEP_APPWORLD_OUTPUTS", "1")
    assert real_appworld.sweep_experiment_outputs("run_B") == 0
    assert (root / "run_B_z").is_dir()


@appworld
def test_appworld_test_splits_are_locally_evaluable():
    """The held-out test splits ship test checkers -> AppWorld.evaluate() works (minimal mode).

    Asserts the env opens a ``test_normal`` task and scores an (empty) trajectory with a real test
    count, i.e. ``evaluate()`` did not crash on missing reference-solution ground truth.
    """
    from tts.env.real_appworld import AppWorldTaskEnv
    from tts.trajectory import Trajectory

    for split in ("test_normal", "test_challenge"):
        env = AppWorldTaskEnv(split=split, level=1, index=0)
        ri = env.evaluate(Trajectory(steps=[]))  # nothing done -> should fail, but be SCORED
        assert ri.grounded and ri.success is False
        assert "AppWorld.evaluate TGC" in ri.reason and "/" in ri.reason
        env.close()


@appworld
@needs_llm
def test_appworld_b0_real_react_agent_runs():
    """B0 drives the official ReAct code agent against the real env and scores via evaluate()."""
    from tts.env.real_appworld import AppWorldTaskEnv

    env = AppWorldTaskEnv(split="dev", level=1, index=0)
    cfg = Config(method="b0", env="appworld", N=1, seed=0,
                 solver=SolverConfig(agent="llm", max_steps=20))
    out = run_decision(env, cfg)
    r = out.record
    assert r.n_commits == 1 and r.simulator_writes == 0
    assert r.reward in (0.0, 1.0)  # official Task Goal Completion
    env.close()
