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


class _StubPredictor:
    """Deterministic stand-in for the LLM world-model predictor (keeps tau2 WM tests offline).

    Predicts SUCCESS iff the candidate action sequence equals the gold sequence, and returns a
    canned observation.  The agentic methods proactively READ via the reader (so the multi-turn
    DB-query path is exercised) before predicting — no LLM, no execution."""

    def __init__(self) -> None:
        self.success_calls = 0
        self.obs_calls = 0
        self.reads = 0

    @staticmethod
    def _scan(args):
        out = []
        for v in (args or {}).values():
            if isinstance(v, (str, int)) and not isinstance(v, bool):
                out.append(str(v))
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        out += [str(z) for z in x.values()
                                if isinstance(z, (str, int)) and not isinstance(z, bool)]
        return out

    def _query(self, reader, entity_types, actions):
        refs = [(t, v) for a in actions for v in self._scan(a.args) for t in entity_types]
        if refs:
            rows = reader.query(refs[:20], "stub")
            self.reads += len(refs[:20])
            return rows
        return {}

    # one-shot (flat / B1)
    def predict_success(self, facts, candidate_actions, gold_actions, tool_docs="") -> bool:
        self.success_calls += 1
        return ([a.name for a in candidate_actions] == [a.name for a in gold_actions]
                and [a.args for a in candidate_actions] == [a.args for a in gold_actions])

    def predict_observation(self, facts, history, action, tool_doc=""):
        self.obs_calls += 1
        return (f"stub::{action.name}", False)

    # agentic, multi-turn (Ours)
    def predict_success_agentic(self, domain, candidate_actions, gold_actions, reader, entity_types,
                                tool_docs="", **kwargs) -> bool:
        self.success_calls += 1
        self._query(reader, entity_types, list(candidate_actions) + list(gold_actions))
        return ([a.name for a in candidate_actions] == [a.name for a in gold_actions]
                and [a.args for a in candidate_actions] == [a.args for a in gold_actions])

    def predict_observation_agentic(self, domain, action, history, reader, entity_types,
                                    tool_doc="", **kwargs):
        self.obs_calls += 1
        self._query(reader, entity_types, [action])
        return (f"stub::{action.name}", False)


def _install_stub_predictor(monkeypatch) -> _StubPredictor:
    from tts.env import real_tau2

    stub = _StubPredictor()
    monkeypatch.setattr(real_tau2, "_build_wm_predictor", lambda domain: stub)
    return stub


@tau2
def test_tau2_official_gold_trajectory_scores_one():
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    gold = env.gold_trajectory()
    ri = env.evaluate(gold)
    assert ri.grounded and ri.success and ri.reward == 1.0  # official ENV evaluation


@tau2
def test_tau2_gold_trajectory_includes_required_communication():
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    criteria = env._task.evaluation_criteria
    required = list(getattr(criteria, "communicate_info", None) or [])
    final_message = env.gold_trajectory().final_message
    assert all(str(value) in final_message for value in required)


@tau2
def test_tau2_world_model_never_writes_real_db(monkeypatch):
    from tts.env.registry import make_env

    _install_stub_predictor(monkeypatch)
    env = make_env("tau2", index=0)
    pre = env.state_hash()
    cfg = Config(method="ours", env="tau2", N=4, seed=0, solver=SolverConfig(p_correct=0.5),
                 simulator=SimulatorConfig(recursion_mode="uncertainty", tau=0.9, node_read_limit=1))
    out = run_decision(env, cfg)
    assert out.record.simulator_writes == 0
    assert out.record.n_commits == 1


@tau2
def test_tau2_agentic_world_model_queries_db(monkeypatch):
    """Ours is a multi-turn read loop: the world-model subagent proactively QUERIES the DB (via the
    read-only reader) before predicting — reads are logged in the spawn tree, the real DB is never
    written, and no environment is executed."""
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.env.registry import make_env

    stub = _install_stub_predictor(monkeypatch)
    env = make_env("tau2", index=0)
    pre = env.state_hash()
    wm = env.world_model(
        SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9,
                        node_read_limit=2, hard_cap=8), Budget(), ReadCache())
    ri = wm.evaluate(env.gold_trajectory())
    assert wm.spawn_root is not None and wm.spawn_root.total_reads() > 0  # it queried the DB
    assert stub.reads > 0 and stub.success_calls >= 1
    assert env.state_hash() == pre  # querying never wrote the real DB


@tau2
@pytest.mark.parametrize("domain", ["airline", "retail", "mock"])
def test_tau2_simulator_is_read_only(domain, monkeypatch):
    """The world model can only READ the backing DB — it can never mutate it.

    Asserted three ways: (1) the ``EntityStore`` exposes no write API; (2) privileged reads are
    deep-copied, so tampering with a read result cannot leak into the real DB; (3) running the
    write-heavy ``ours`` decision leaves the real DB hash untouched (``simulator_writes == 0``)."""
    from tts.abcs import EntityStore
    from tts.env.datasets import TaskSpec
    from tts.env.registry import _tau2_usable_tasks, make_env_from_spec

    _install_stub_predictor(monkeypatch)

    # (1) structural: no mutating method on the read-only store interface
    api = [m for m in dir(EntityStore) if not m.startswith("_")]
    assert not any(k in m for m in api for k in ("write", "set", "update", "commit", "delete", "put"))

    if not _tau2_usable_tasks(domain):
        pytest.skip(f"no world-model-eligible {domain} tasks")
    env = make_env_from_spec(TaskSpec("tau2", domain, 0, domain=domain), usable_only=True)

    # (2) deep-copy isolation: tampering with a read result must not change a fresh read or the DB
    pre = env.state_hash()
    store = env.entity_store()
    btype = next(iter(env._model().buckets))
    bid = store.list_ids(btype)[0]
    snapshot = store.read(btype, bid)["model"].model_dump()
    tampered = store.read(btype, bid)["model"]
    tampered.__dict__[next(iter(tampered.__dict__))] = "__TAMPERED__"
    assert store.read(btype, bid)["model"].model_dump() == snapshot  # fresh read is pristine
    assert env.state_hash() == pre

    # (3) a full write-heavy decision never writes the real env during sampling (only the single
    # commit may write); the simulator phase is asserted clean via simulator_writes == 0.
    cfg = Config(method="ours", env="tau2", N=4, seed=0, solver=SolverConfig(p_correct=0.5),
                 simulator=SimulatorConfig(recursion_mode="uncertainty", tau=0.9, node_read_limit=1))
    out = run_decision(env, cfg)
    assert out.record.simulator_writes == 0
    assert out.record.n_commits == 1


@tau2
def test_tau2_read_action_returns_real_retrieval(monkeypatch):
    """A read/search action's observation is the EXACT result the real env returns — the world
    model retrieves it the same way the official evaluation does (read-only, no mutation)."""
    from tau2.data_model.message import ToolCall
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.env.registry import make_env

    _install_stub_predictor(monkeypatch)
    env = make_env("tau2", index=0)  # airline
    read_tools = set(env._read_tool_names())
    ra = next((a for a in env.gold_trajectory().actions if a.name in read_tools), None)
    if ra is None:
        pytest.skip("no read action in gold trajectory")
    pre = env.state_hash()
    wm = env.world_model(SimulatorConfig(kind="recursive", node_read_limit=2), Budget(), ReadCache())
    obs = wm.step(ra)
    real = env.fork()._env.get_response(
        ToolCall(id="x", name=ra.name, requestor="assistant", arguments=dict(ra.args)))
    assert obs.content == str(getattr(real, "content", real))  # retrieved the same way as the env
    assert env.state_hash() == pre  # read-only: never mutated the real DB


@tau2
@pytest.mark.parametrize("domain", ["airline", "retail", "mock"])
def test_tau2_world_model_predicts_without_constructing_envs(domain, monkeypatch):
    """The world model never constructs an environment and never writes the real DB.

    With ``_make_environment`` patched to raise, the WM can still step() (read-only tool actions are
    served directly from the real env; write actions are predicted) and evaluate() — proving it
    constructs no env (real or forked) — while the real DB hash stays unchanged and the predictor is
    consulted for the success verdict."""
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.env import real_tau2
    from tts.env.datasets import TaskSpec
    from tts.env.registry import _tau2_usable_tasks, make_env_from_spec

    if not _tau2_usable_tasks(domain):
        pytest.skip(f"no world-model-eligible {domain} tasks")
    stub = _install_stub_predictor(monkeypatch)
    env = make_env_from_spec(TaskSpec("tau2", domain, 0, domain=domain), usable_only=True)
    read_tools = set(env._read_tool_names())
    pre = env.state_hash()
    wm = env.world_model(
        SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9,
                        node_read_limit=2, hard_cap=8),
        Budget(), ReadCache(),
    )

    # after the env exists, the WM must construct NO further environment
    monkeypatch.setattr(real_tau2, "_make_environment",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("WM built an env")))

    gold = env.gold_trajectory()
    obs = [wm.step(a) for a in gold.actions]
    ri = wm.evaluate(gold)                             # predicts success, constructs nothing
    # read-only actions are served from the real env; write actions come from the (stub) predictor
    for a, o in zip(gold.actions, obs):
        if a.name not in read_tools:
            assert str(o.content).startswith("stub::")  # write action -> predicted
    assert stub.success_calls >= 1 and ri.success and not ri.grounded  # gold == gold -> success
    assert env.state_hash() == pre                     # nothing was written to the real DB


@tau2
@needs_llm
def test_tau2_llm_world_model_predicts_gold_success():
    """Smoke (real LLM): the LLM world model predicts the gold trajectory succeeds, no execution."""
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    pre = env.state_hash()
    wm = env.world_model(
        SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9,
                        node_read_limit=1, hard_cap=8),
        Budget(), ReadCache(),
    )
    ri = wm.evaluate(env.gold_trajectory())
    assert ri.success and not ri.grounded
    assert env.state_hash() == pre


@tau2
@pytest.mark.parametrize("domain", ["airline", "retail", "mock"])
def test_tau2_single_control_domains_world_model(domain, monkeypatch):
    """Every single-control domain: gold scores 1.0 officially and the WM predicts success for it.

    Exercises the domain-generic read-grounded world model (introspected DB buckets + reference
    closure) over airline (reservations), retail (orders) and mock (tasks)."""
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.env.datasets import TaskSpec
    from tts.env.registry import _tau2_usable_tasks, make_env_from_spec

    if not _tau2_usable_tasks(domain):
        pytest.skip(f"no world-model-eligible {domain} tasks")
    _install_stub_predictor(monkeypatch)

    env = make_env_from_spec(TaskSpec("tau2", domain, 0, domain=domain), usable_only=True)
    assert env.env_name == f"tau2_{domain}"

    gold = env.gold_trajectory()
    ri = env.evaluate(gold)
    assert ri.grounded and ri.success and ri.reward == 1.0  # official ENV evaluation (real)

    pre = env.state_hash()
    rec = env.world_model(
        SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9,
                        node_read_limit=1, hard_cap=8),
        Budget(), ReadCache(),
    ).evaluate(gold)
    assert rec.success and rec.reward == 1.0 and not rec.grounded  # predicted (not grounded)
    assert env.state_hash() == pre  # the world model never wrote the real DB


@tau2
def test_tau2_mutating_tools_derived_from_toolkit():
    """Per-domain mutating tools are read off the official toolkit (ToolType.WRITE)."""
    from tts.env.real_tau2 import tau2_mutating_tools

    air = tau2_mutating_tools("airline")
    assert {"book_reservation", "cancel_reservation"} <= set(air)
    assert tau2_mutating_tools("retail")  # retail has its own WRITE tools (cancel/modify/return...)
    assert tau2_mutating_tools("airline") != tau2_mutating_tools("retail")


@tau2
@pytest.mark.parametrize("domain", ["telecom", "telecom-workflow", "banking_knowledge"])
def test_tau2_agentic_world_model_covers_all_domains(domain, monkeypatch):
    """The agentic world model now applies to dual-control (telecom) and retrieval (banking)
    domains too: it reads records / lists ids / calls read-only tools, predicts, and never writes."""
    from tts.budget import Budget
    from tts.cache import ReadCache
    from tts.env.datasets import TaskSpec
    from tts.env.registry import _tau2_usable_tasks, make_env_from_spec

    if not _tau2_usable_tasks(domain):
        pytest.skip(f"no world-model-eligible {domain} tasks (env/deps unavailable)")
    stub = _install_stub_predictor(monkeypatch)
    env = make_env_from_spec(TaskSpec("tau2", domain, 0, domain=domain), usable_only=True)
    assert env.env_name == f"tau2_{domain}"
    pre = env.state_hash()
    wm = env.world_model(
        SimulatorConfig(kind="recursive", recursion_mode="uncertainty", tau=0.9,
                        node_read_limit=2, hard_cap=8), Budget(), ReadCache())
    ri = wm.evaluate(env.gold_trajectory())
    assert wm.spawn_root is not None and wm.spawn_root.total_reads() > 0  # it queried the DB
    assert stub.success_calls >= 1
    assert env.state_hash() == pre  # querying never wrote the real DB


@tau2
def test_tau2_read_tool_runner_is_read_only(monkeypatch):
    """The world model may CALL read-only tools to retrieve info (search/KB), but the runner
    refuses any non-READ tool and never mutates the real DB."""
    from tts.env.registry import make_env

    env = make_env("tau2", index=0)
    pre = env.state_hash()
    read_tools = env._read_tool_names()
    assert "search_direct_flight" in read_tools  # a read-only retrieval tool is exposed
    # a read-only call returns data without mutating the DB
    out = env._run_read_tool("list_all_airports", {})
    assert isinstance(out, str) and env.state_hash() == pre
    # a WRITE tool is refused by the runner (structural write-safety)
    with pytest.raises(PermissionError):
        env._run_read_tool("cancel_reservation", {"reservation_id": "XEHM4B"})
    assert env.state_hash() == pre


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


def test_appworld_open_world_only_attempts_minimal_mode(monkeypatch):
    """A minimal-mode failure must be reported directly, not masked by a full-mode retry."""
    import sys
    from types import ModuleType

    from tts.env import appworld_lowlevel, real_appworld

    attempts = []
    expected = RuntimeError("minimal initialization failed")

    class FakeAppWorld:
        def __init__(self, *, task_id, experiment_name, ground_truth_mode):
            attempts.append((task_id, experiment_name, ground_truth_mode))
            raise expected

    fake_appworld = ModuleType("appworld")
    fake_appworld.AppWorld = FakeAppWorld
    monkeypatch.setitem(sys.modules, "appworld", fake_appworld)
    monkeypatch.setattr(appworld_lowlevel, "_ensure_root", lambda root: None)

    with pytest.raises(RuntimeError, match="minimal initialization failed") as raised:
        real_appworld._open_world("task_1", "experiment")

    assert raised.value is expected
    assert attempts == [("task_1", "experiment", "minimal")]


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
