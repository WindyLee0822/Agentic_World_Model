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
import os
import shutil
import threading
import uuid
from typing import List, Optional

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

        with _world_session(self._task_id, self._experiment_base, f"s{seed}") as world:
            code_blocks, outputs = run_react_code_agent(
                world, chat, max_steps=max_steps, temperature=solver.temperature,
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
