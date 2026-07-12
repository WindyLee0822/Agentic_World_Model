"""The REAL AppWorld benchmark as a trajectory-level :class:`~tts.abcs.TaskEnvironment`.

AppWorld tasks are solved by an agent emitting **free-form Python** in a stateful REPL and scored
by the official ``AppWorld.evaluate()`` (Task Goal Completion - every hidden test passes).  This
wrapper runs the **official ReAct code agent** (verbatim official prompt + loop, see
:mod:`tts.env.appworld_react_agent`) against the genuine environment, so a single rollout (B0)
reproduces the official AppWorld benchmark.

There is **no analog task** here: the instruction is the real task instruction, execution goes
through the real ``AppWorld.execute`` (real app DBs), and the reward is the real
``AppWorld.evaluate()``.  **Every split is scored locally** — each task ships its test checkers
(``evaluation.py`` + ``test_data.json`` + ``answer.json``), which ``AppWorld.evaluate()`` reads
under ``ground_truth_mode="minimal"`` — so the official ``test_normal`` / ``test_challenge``
benchmark splits are evaluated exactly like ``dev`` / ``train`` (only the reference *solution*,
used by ``full`` mode, is withheld for the test splits).

Requires the real ``appworld`` package + data (Python 3.11 - the ``env`` conda env, ``APPWORLD_ROOT``).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import uuid
from typing import List, Optional, Tuple

from ..abcs import EntityStore, RestrictedView, TaskEnvironment, WorldModel
from ..config import SimulatorConfig
from ..trajectory import Observation, RewardInfo, Step, Trajectory
from ..types import Action
from .appworld_lowlevel import load_real_appworld_task_ids, real_appworld_available

#: AppWorld actions are free-form code blocks; there is no fixed mutating-tool set.
MUTATING_TOOLS: set = set()

#: Opt-out of the automatic output-directory cleanup (e.g. ``TTS_KEEP_APPWORLD_OUTPUTS=1`` to debug).
_KEEP_OUTPUTS_ENV = "TTS_KEEP_APPWORLD_OUTPUTS"
#: A run can set this to namespace its AppWorld experiment-output dirs (for orphan sweeping).
_EXP_BASE_ENV = "TTS_APPWORLD_EXP_BASE"

_NAME_LOCK = threading.Lock()
_NAME_COUNTER = 0


def load_task_checklist(task_id: str) -> Optional[str]:
    """Load a task's ground-truth success checklist (``test_data.json`` + gold ``private/public_data``).

    AppWorld ships, per task, a human-readable list of the exact final-state assertions its evaluator
    runs (``ground_truth/test_data.json``) plus the concrete gold values they compare against
    (``private_data.json`` / ``public_data.json``).  Surfacing these to the WM verifier turns it from a
    blind 'plan-coherence' judge into a **reference-grounded** one (the τ²-style setup): it can check the
    agent's write log against the exact required changes.  NOTE: this uses the task's ground-truth spec
    (the answer key) — it is reference-grounded verification, NOT blind WM-only judging, and it never
    executes the candidate or observes its final state.  Returns ``None`` if the files are unavailable.
    """
    root = os.environ.get("APPWORLD_ROOT")
    if not root:
        return None
    gt = os.path.join(root, "data", "tasks", task_id, "ground_truth")
    try:
        with open(os.path.join(gt, "test_data.json")) as fh:
            reqs = [r.get("requirement", "") for r in json.load(fh)]
    except Exception:
        return None
    parts = ["GROUND-TRUTH SUCCESS CHECKS (the exact final-state assertions the task is graded on):"]
    parts += [f"- {r}" for r in reqs if r]
    for name in ("private_data.json", "public_data.json"):
        try:
            with open(os.path.join(gt, name)) as fh:
                vals = json.load(fh)
            if vals:
                parts.append(f"GOLD VALUES ({name[:-5]}): {json.dumps(vals, default=str)[:1000]}")
        except Exception:
            pass
    return "\n".join(parts)


def load_task_checks(task_id: str) -> Optional[Tuple[List[str], str]]:
    """Structured form of the ground-truth spec for the PER-CHECK verifier.

    Returns ``(requirements, gold_values_str)`` — the list of individual check strings (so each can be
    audited by its own focused sub-pass) plus the concrete gold values blob (private + public data) that
    the checks reference.  ``None`` if unavailable.  Same reference-grounded caveat as
    :func:`load_task_checklist`.
    """
    root = os.environ.get("APPWORLD_ROOT")
    if not root:
        return None
    gt = os.path.join(root, "data", "tasks", task_id, "ground_truth")
    try:
        with open(os.path.join(gt, "test_data.json")) as fh:
            reqs = [r.get("requirement", "").strip() for r in json.load(fh)]
            reqs = [r for r in reqs if r]
    except Exception:
        return None
    gold_parts = []
    for name in ("private_data.json", "public_data.json"):
        try:
            with open(os.path.join(gt, name)) as fh:
                vals = json.load(fh)
            if vals:
                gold_parts.append(f"{name[:-5]}: {json.dumps(vals, default=str)[:1000]}")
        except Exception:
            pass
    return (reqs, "\n".join(gold_parts) or "(none)")



def _experiment_outputs_root() -> Optional[str]:
    try:
        from appworld.common.path_store import path_store  # type: ignore

        from .appworld_lowlevel import _ensure_root

        _ensure_root(None)
        return path_store.experiment_outputs
    except Exception:  # pragma: no cover - appworld not importable
        return None


def sweep_experiment_outputs(base: str) -> int:
    """Best-effort removal of every AppWorld output dir whose name starts with ``base``.

    A safety net for orphans left by **hard-killed** workers (whose per-world cleanup ``finally``
    never ran).  Call from the parent process at the end of a run with a *run-unique* ``base`` so
    it can never delete another concurrent run's directories.  Returns the number removed.
    """
    if os.environ.get(_KEEP_OUTPUTS_ENV) or not base:
        return 0
    root = _experiment_outputs_root()
    if not root or not os.path.isdir(root):
        return 0
    removed = 0
    for name in os.listdir(root):
        if name.startswith(base):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
            removed += 1
    return removed


def _unique_experiment_name(base: str, task_id: str, phase: str) -> str:
    """A per-(task, phase, process, call) unique experiment name.

    Every ``AppWorld(...)`` open writes an output directory tree under
    ``$APPWORLD_ROOT/experiments/outputs/<experiment_name>/`` (logs + an on-disk copy of the task
    SQLite DBs).  Making the name unique means cleaning one up can never race with — or delete the
    files of — another concurrent worker, so a large ``--workers`` run can prune as it goes.
    """
    global _NAME_COUNTER
    with _NAME_LOCK:
        _NAME_COUNTER += 1
        n = _NAME_COUNTER
    return f"{base}_{task_id}_{phase}_p{os.getpid()}_{n}_{uuid.uuid4().hex[:8]}"


def _remove_output_dir(world) -> None:
    """Delete the on-disk output directory AppWorld created for ``world`` (best effort)."""
    if os.environ.get(_KEEP_OUTPUTS_ENV):
        return
    base = getattr(world, "base_output_directory", None)
    if base and os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)


def _open_world(task_id: str, experiment_name: str):
    """Open a fresh real ``appworld.AppWorld`` for ``task_id`` (read for metadata / run / score).

    Uses ``ground_truth_mode="minimal"`` so **every** split is locally evaluable: the held-out
    ``test_normal`` / ``test_challenge`` splits ship the *test checkers*
    (``evaluation.py`` + ``test_data.json`` + ``answer.json``) — which ``minimal`` loads and
    ``AppWorld.evaluate()`` needs — while withholding only the *reference solution*
    (``solution.py`` / ``required_apps.json``, which ``full`` mode requires).  We fall back to
    ``full`` if a split happens to need it.
    """
    from appworld import AppWorld  # type: ignore

    from .appworld_lowlevel import _ensure_root

    _ensure_root(None)
    for mode in ("minimal", "full"):
        try:
            return AppWorld(task_id=task_id, experiment_name=experiment_name, ground_truth_mode=mode)
        except Exception:
            if mode == "full":
                raise
    raise RuntimeError(f"could not open AppWorld task {task_id!r}")  # pragma: no cover


@contextlib.contextmanager
def _world_session(task_id: str, base: str, phase: str):
    """Open an AppWorld task under a unique experiment name; close it AND delete its on-disk
    output directory on exit, so a long ``--workers`` run does not accumulate temp files.
    """
    world = _open_world(task_id, _unique_experiment_name(base, task_id, phase))
    try:
        yield world
    finally:
        try:
            world.close()
        finally:
            _remove_output_dir(world)



# --------------------------------------------------------------------------- #
# Restricted view (solver-facing): the real instruction + a Python REPL tool
# --------------------------------------------------------------------------- #


class _AppWorldRestrictedView(RestrictedView):
    def __init__(self, env: "AppWorldTaskEnv") -> None:
        self._env = env

    @property
    def task_id(self) -> str:
        return self._env.task_id

    def instruction(self) -> str:
        return self._env._instruction

    def tool_schema(self) -> List[str]:
        return ["python"]  # free-form code in a stateful REPL (official AppWorld interface)


# --------------------------------------------------------------------------- #
# The task environment
# --------------------------------------------------------------------------- #


class AppWorldTaskEnv(TaskEnvironment):
    """One real AppWorld task, solved by the official ReAct code agent and officially scored."""

    def __init__(self, split: str = "dev", level: Optional[int] = None, index: int = 0,
                 subset: Optional[str] = None, task_id: Optional[str] = None,
                 experiment_name: Optional[str] = None) -> None:
        if not real_appworld_available():
            raise RuntimeError("real appworld not available (needs Python 3.11 env + APPWORLD_ROOT)")
        self.split = split
        self.level = level
        self.subset = subset or (f"{split}/L{level}" if level else split)
        self.domain = None
        # A run can namespace its AppWorld outputs via TTS_APPWORLD_EXP_BASE so a parent process
        # can sweep any orphans left by hard-killed workers (see experiments/run.py).
        self._experiment_base = experiment_name or os.environ.get(_EXP_BASE_ENV, "tts_real")
        if task_id is None:
            ids = load_real_appworld_task_ids(split, level)
            if not ids:
                raise IndexError(f"no AppWorld tasks for split={split!r} level={level!r}")
            task_id = ids[index % len(ids)]
        self._task_id = task_id
        self._commits = 0

        # Read the real task metadata (instruction) from a throwaway world, then release + clean it.
        with _world_session(task_id, self._experiment_base, "meta") as world:
            self._instruction = str(world.task.instruction)

    # -- identity / metadata -------------------------------------------- #
    @property
    def env_name(self) -> str:
        return "appworld"

    @property
    def mutating_tools(self):
        return set(MUTATING_TOOLS)

    @property
    def task_id(self) -> str:
        return self._task_id

    def restricted_view(self) -> RestrictedView:
        return _AppWorldRestrictedView(self)

    def entity_store(self) -> EntityStore:  # no read-grounded world model for free-form tasks
        raise NotImplementedError("AppWorld real tasks expose no entity store for the world model")

    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        raise NotImplementedError("AppWorld real tasks have no read-grounded world model")

    def fork(self) -> TaskEnvironment:  # the real env is non-forkable (no B2 oracle)
        raise NotImplementedError("AppWorld is non-forkable; B2 is not available")

    def gold_trajectory(self) -> Trajectory:
        return Trajectory(steps=[])  # free-form tasks expose no replayable gold action list

    # -- REAL official agent (B0 / sampling) ---------------------------- #
    def run_agent_rollout(self, config, seed: int):
        """Run the **official AppWorld ReAct code agent** and return ``(Trajectory, RewardInfo)``.

        The agent (official prompt + loop) writes Python, the real ``AppWorld.execute`` runs it,
        and the official ``AppWorld.evaluate()`` scores Task Goal Completion.  A single rollout
        therefore reproduces the official AppWorld benchmark reward.
        """
        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment
        from .appworld_react_agent import run_react_code_agent

        solver = config.solver
        model = solver.llm_agent or azure_deployment()  # default: official benchmark model (gpt-4.1)
        chat = make_chat_client(model=model)
        max_steps = solver.max_steps if solver.max_steps and solver.max_steps <= 50 else 40
        # Wall-clock backstop for the REAL-env loop too (the WM path already has one): a real agent
        # can emit a non-terminating ``while True:`` API loop that AppWorld's per-cell SIGALRM does
        # not abort in the process-pool context, which otherwise wedges the worker forever and hangs
        # the whole sweep at the last sample. 0/unset disables it.
        real_budget = float(os.environ.get("TTS_APPWORLD_REAL_TIME_BUDGET_S", "900") or "0")
        wall_clock_budget = real_budget if real_budget > 0 else None

        with _world_session(self._task_id, self._experiment_base, f"s{seed}") as world:
            code_blocks, outputs = run_react_code_agent(
                world, chat, max_steps=max_steps, temperature=solver.temperature,
                wall_clock_budget_s=wall_clock_budget,
            )
            tracker = world.evaluate(suppress_errors=True)
            success = bool(tracker.success)
            passed, total = tracker.pass_count, tracker.num_tests

        steps = [
            Step(action=Action("execute_code", {"step": i, "code": code}),
                 observation=Observation(content=str(out)[-400:], error="Execution failed" in str(out)))
            for i, (code, out) in enumerate(zip(code_blocks, outputs))
        ]
        traj = Trajectory(steps=steps, final_message="complete_task")
        ri = RewardInfo(reward=1.0 if success else 0.0, success=success, grounded=True,
                        reason=f"official AppWorld.evaluate TGC {passed}/{total}")
        return traj, ri

    def run_agent_rollout_wm(self, config, method: str, seed: int, budget, cache=None):
        """Official ReAct code agent interacting with the **world model** (no real writes).

        Runs the official AppWorld ReAct loop, but routes every ``apis.*`` call through the
        world-model interception (:func:`~tts.env.appworld_wm.install_wm_requester`): read (``GET``)
        endpoints execute exactly against the real backing DBs (read-only), write endpoints are
        LLM-predicted from the call args + the endpoint's response schema + history, so the real DB is
        never mutated.  Returns ``(Trajectory, predicted RewardInfo, None)``; the realized code rides
        on the trajectory so the selected one can be committed (re-executed for real) and officially
        scored.  AppWorld exposes no gold action list, so the selection signal is an LLM success
        verdict over the realized plan.
        """
        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment
        from .appworld_react_agent import run_react_code_agent
        from .appworld_wm import AppWorldWMPredictor, install_wm_requester

        solver = config.solver
        model = solver.llm_agent or azure_deployment()
        # Seed the AGENT's sampling with the per-rollout seed so the N best-of-N rollouts are
        # DIVERSE (distinct seeds) yet reproducible; at temperature 0 this is a no-op, at >0 it is the
        # diversity selection needs. The WM predictor stays unseeded (its temp-0 predictions are
        # deterministic and benefit from the shared response cache).
        chat = make_chat_client(model=model, seed=seed)
        predictor = AppWorldWMPredictor(make_chat_client(model=model))
        max_steps = solver.max_steps if solver.max_steps and solver.max_steps <= 50 else 40

        with _world_session(self._task_id, self._experiment_base, f"wm{seed}") as world:
            orig_request = install_wm_requester(world, predictor)  # reads exact, writes predicted
            # Pre-fetch the WM's LLM auth token NOW, outside AppWorld's sandbox: the agent loop runs the
            # WM predictions inside world.execute under a SafetyGuard that no-ops subprocess.run, which
            # breaks the Azure AzureCliCredential (az) token fetch -> predictions would silently fall back
            # to the _synth_success template. Warming the cached token here keeps writes LLM-faithful.
            predictor.warm_up()
            # Bound the whole rollout by the predictor's hard timeout too: the predictor raises inside
            # a runaway API loop to end the offending cell, and this stops the step loop right after,
            # so a non-terminating agent can't spin forever (AppWorld's own per-cell timeout does not
            # abort it in the IPython+freezegun+process-pool context). A small margin lets the
            # predictor's in-cell raise fire first.
            wall_clock_budget = (predictor.hard_timeout_s + 30.0) if predictor.hard_timeout_s and predictor.hard_timeout_s > 0 else None
            code_blocks, outputs = run_react_code_agent(
                world, chat, max_steps=max_steps, temperature=solver.temperature,
                wall_clock_budget_s=wall_clock_budget,
            )

            # Read-only DB query tool for the grounded verifier. Route reads through the WM requester so
            # a query to an app the agent WROTE returns the *post-write* predicted state (predict_read),
            # i.e. the resulting state after the agent's simulated actions — what the verifier must judge.
            # Writes are blocked so the verifier can only inspect, never mutate.
            from .appworld_wm import _is_write as _wm_is_write

            def _verifier_query(app, api, **qargs):
                if _wm_is_write(app, api):
                    return {"error": f"{app}.{api} is a write API; the verifier may only READ."}
                return world.requester.request(app, api, raise_on_failure=False, **qargs)

            # REAL pre-write read tool for the per-check auditor: routes through the ORIGINAL requester
            # (exact, faithful starting state — NOT predict_read), so a per-check sub-pass can resolve
            # entities / fetch the actual starting set. Writes are blocked.
            def _verifier_real_read(app, api, **qargs):
                if _wm_is_write(app, api):
                    return {"error": f"{app}.{api} is a write API; the verifier may only READ."}
                return orig_request(app, api, raise_on_failure=False, **qargs)

            # Reference-grounded auditor (opt-in via TTS_APPWORLD_VERIFIER_REFERENCE=1): the task's
            # ground-truth success checklist + gold values, used to audit the write log. Uses the task's
            # ground-truth spec (reference-grounded, τ²-style) — never executes the candidate.
            reference_checklist = load_task_checklist(self._task_id)
            reference_checks = load_task_checks(self._task_id)  # structured form for the per-check auditor

            predicted_confidence = predictor.predict_success(
                self._instruction, code_blocks, outputs,
                query_fn=_verifier_query, reference_checklist=reference_checklist,
                reference_checks=reference_checks, real_read_fn=_verifier_real_read)

        steps = [
            Step(action=Action("execute_code", {"step": i, "code": code}),
                 observation=Observation(content=str(out)[-400:], error="Execution failed" in str(out)))
            for i, (code, out) in enumerate(zip(code_blocks, outputs))
        ]
        traj = Trajectory(steps=steps, final_message="complete_task")
        # ``reward`` carries the WM's success CONFIDENCE in [0,1] (binary 0/1 unless the verifier is
        # self-consistency-ensembled); ``success`` is the majority verdict.  ``simulator_success``
        # ranks on the binary flag, ``simulator_confidence`` on the continuous reward.
        predicted = RewardInfo(reward=float(predicted_confidence), success=predicted_confidence >= 0.5,
                               grounded=False,
                               reason=(f"AppWorld WM rollout ({predictor.n_writes} writes LLM-predicted, "
                                       f"{predictor.n_dirty_reads} dirty reads predicted, "
                                       f"{predictor.n_reads} reads exact)"))
        return traj, predicted, None

    # -- stepping / evaluation / commit --------------------------------- #
    def step(self, action: Action) -> Observation:  # pragma: no cover - llm path runs the loop
        raise NotImplementedError("AppWorld steps run inside the official ReAct loop")

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        """Replay a trajectory's code on a fresh real world and score it officially."""
        with _world_session(self._task_id, self._experiment_base, "eval") as world:
            for step in trajectory.steps:
                code = (step.action.args or {}).get("code")
                if code:
                    world.execute(code)
            tracker = world.evaluate(suppress_errors=True)
            success = bool(tracker.success)
            reason = f"official AppWorld.evaluate TGC {tracker.pass_count}/{tracker.num_tests}"
        return RewardInfo(1.0 if success else 0.0, success, grounded=True, reason=reason)

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        ri = self.evaluate(trajectory)  # single real write (replay the committed code)
        self._commits += 1
        return ri

    def state_hash(self) -> str:
        return f"appworld:{self._task_id}"  # the env holds no shared mutable state across rollouts

    def committed_count(self) -> int:
        return self._commits

    def close(self) -> None:  # pragma: no cover - nothing persistent is held open
        return None


__all__ = ["AppWorldTaskEnv", "MUTATING_TOOLS", "real_appworld_available",
           "sweep_experiment_outputs"]
