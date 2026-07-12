"""The REAL OfficeBench benchmark as a trajectory-level :class:`~tts.abcs.TaskEnvironment`.

OfficeBench tasks are solved by an agent that emits JSON office-app actions
(``{'app','action',...}``) against a Docker ``/testbed`` filesystem, and scored by execution-based,
file-state checks (``utils/evaluate.py``).  This wrapper runs the **official OfficeBench agent loop**
(:mod:`tts.env.officebench_agent`) against a throwaway per-rollout container and scores it with the
**official evaluation config**, so a single rollout (B0) reproduces the official OfficeBench number.

Like AppWorld (free-form actions, execution-based scoring, **no replayable gold action list**), there
is no read-grounded DB-replay world model: B1/Ours fall back to the oracle reward on the real-agent
path, and the ``llm_wm`` path uses an LLM verifier (:class:`tts.env.officebench_wm.OfficeBenchVerifier`)
as the selection signal, then commits (re-executes) the selected trajectory and scores it officially.

Each rollout materialises a fresh ``/testbed`` from the task's initial files in its own container, so
the env holds **no shared mutable state across rollouts** (``state_hash`` is constant) and a
world-model/shadow rollout can never mutate another rollout's — or the committed — state.

Requires the OfficeBench clone (``OFFICEBENCH_ROOT``) + a usable Docker daemon (the app/eval image is
built on demand from ``officebench_docker/``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import os

from ..abcs import EntityStore, RestrictedView, TaskEnvironment, WorldModel
from ..config import SimulatorConfig
from ..trajectory import Observation, RewardInfo, Step, Trajectory
from ..types import Action
from .datasets import parse_officebench_subset
from .officebench_lowlevel import (
    OfficeBenchSession,
    load_officebench_subtasks,
    load_subtask_config,
    real_officebench_available,
)

#: OfficeBench actions are free-form office-app calls; there is no fixed mutating-tool set.
MUTATING_TOOLS: set = set()

#: The official OfficeBench apps (for the restricted-view tool schema, without opening a container).
OFFICEBENCH_APPS = ("calendar", "excel", "ocr", "pdf", "shell", "word", "email", "llm")


def _assert_write_safe() -> bool:
    """Whether to runtime-verify the world-model shadow ``/testbed`` stays pristine (default ON).

    When on, an ``llm_wm`` rollout digests the shadow before and after sampling+judging and raises if
    it changed — enforcing that no write (predicted or incidental) reaches the environment before the
    single commit.  Set ``TTS_OFFICEBENCH_ASSERT_WRITE_SAFE=0`` to skip the two extra digest execs.
    """
    return os.environ.get("TTS_OFFICEBENCH_ASSERT_WRITE_SAFE", "1") not in ("0", "false", "False")


def _int_env(name: str, default: int) -> int:
    """A non-negative-int tuning knob (falls back to ``default``)."""
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _scaled_max_steps(level: int) -> int:
    """Default step budget scaled by the task's app-count (Fix 4).

    A fixed 20-step budget starves multi-app OfficeBench tasks — 3app trajectories need to switch
    between apps and author several files, and the ofbench-ours-5.4 run showed a cluster of 3app
    failures hitting the cap mid-task.  This keeps the 1app floor unchanged (20) and adds headroom per
    extra app, capped so a runaway rollout still terminates.  Overridable via the env tunables; an
    explicit ``solver.max_steps`` (≤ cap) still wins over this default.
    """
    floor = _int_env("TTS_OFFICEBENCH_STEPS_FLOOR", 20)
    per_app = _int_env("TTS_OFFICEBENCH_STEPS_PER_APP", 6)
    cap = _int_env("TTS_OFFICEBENCH_STEPS_CAP", 40)
    return min(cap, floor + per_app * max(0, int(level) - 1))


def _action_of(step_action: Dict) -> Dict:
    """Recover the raw OfficeBench action dict stored on a trajectory step."""
    return (step_action.args or {}).get("action") or {}


# --------------------------------------------------------------------------- #
# Restricted view (solver-facing): the real instruction + the office-app tools
# --------------------------------------------------------------------------- #


class _OfficeBenchRestrictedView(RestrictedView):
    def __init__(self, env: "OfficeBenchTaskEnv") -> None:
        self._env = env

    @property
    def task_id(self) -> str:
        return self._env.task_id

    def instruction(self) -> str:
        return self._env._instruction

    def tool_schema(self) -> List[str]:
        return list(OFFICEBENCH_APPS)  # the official office apps the agent may switch between


# --------------------------------------------------------------------------- #
# The task environment
# --------------------------------------------------------------------------- #


class OfficeBenchTaskEnv(TaskEnvironment):
    """One real OfficeBench subtask, solved by the official agent and officially scored."""

    def __init__(self, subset: str = "1app", index: int = 0, *,
                 task_id: Optional[str] = None, subtask_id: Optional[str] = None,
                 root: Optional[str] = None) -> None:
        if not real_officebench_available(root):
            raise RuntimeError(
                "real OfficeBench not available (needs OFFICEBENCH_ROOT clone + a usable Docker daemon)")
        app_count = parse_officebench_subset(subset)
        self.subset = subset
        self.level = app_count
        self.split = None
        self.domain = None
        self._root = root
        if task_id is None or subtask_id is None:
            pairs = load_officebench_subtasks(app_count, root)
            if not pairs:
                raise IndexError(f"no OfficeBench subtasks for {subset!r}")
            task_id, subtask_id = pairs[index % len(pairs)]
        self._task = task_id
        self._subtask = subtask_id
        self._commits = 0

        self._config = load_subtask_config(task_id, subtask_id, root)
        self._instruction = str(self._config.get("task", ""))

    # -- identity / metadata -------------------------------------------- #
    @property
    def env_name(self) -> str:
        return "officebench"

    @property
    def mutating_tools(self):
        return set(MUTATING_TOOLS)

    @property
    def task_id(self) -> str:
        return f"{self._task}/{self._subtask}"

    def restricted_view(self) -> RestrictedView:
        return _OfficeBenchRestrictedView(self)

    def entity_store(self) -> EntityStore:  # free-form file tasks: no read-grounded entity store
        raise NotImplementedError("OfficeBench tasks expose no entity store for the world model")

    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        raise NotImplementedError("OfficeBench tasks have no read-grounded DB-replay world model")

    def fork(self) -> TaskEnvironment:  # B2 falls back to best-of-N over realized rewards (like AppWorld)
        raise NotImplementedError("OfficeBench B2 fork is not available; use real-agent best-of-N")

    def gold_trajectory(self) -> Trajectory:
        return Trajectory(steps=[])  # free-form tasks expose no replayable gold action list

    # -- helpers -------------------------------------------------------- #
    def _new_session(self, phase: str) -> OfficeBenchSession:
        return OfficeBenchSession(self._task, self._subtask, phase=phase, root=self._root)

    def _trajectory_from_run(self, run) -> Trajectory:
        steps: List[Step] = []
        for i, action in enumerate(run.actions):
            obs = run.observations[i] if i < len(run.observations) else ""
            name = self._action_name(action)
            steps.append(Step(
                action=Action(name, {"step": i, "action": action}),
                observation=Observation(content=str(obs)[-400:],
                                        error=str(obs).lower().startswith(("malformed", "failed", "error")))))
        return Trajectory(steps=steps, final_message=run.final_answer or "finish_task")

    @staticmethod
    def _action_name(action: Dict) -> str:
        app = action.get("app", "?")
        act = action.get("action", action.get("raw", "?"))
        return f"{app}.{act}" if app and app != "?" else str(act)

    def _replay(self, session: OfficeBenchSession, trajectory: Trajectory) -> None:
        """Re-execute a recorded trajectory's actions in ``session`` (the single committed write)."""
        workdir = "/"
        for step in trajectory.steps:
            action = _action_of(step.action)
            act = action.get("action")
            if act == "switch_app" or act == "got_stuck" or not action:
                continue  # system bookkeeping: no container effect
            if act == "finish_task":
                session.write_answer(str(action.get("answer", "None")))
                continue
            result = session.exec_action(action, workdir)
            if result.get("is_cd") and result.get("new_workdir"):
                workdir = result["new_workdir"]

    def _score(self, session: OfficeBenchSession) -> RewardInfo:
        res = session.evaluate(self._config.get("evaluation", []))
        success = bool(res.get("passed"))
        reason = f"official OfficeBench evaluate {res.get('num_passed', 0)}/{res.get('num_checks', 0)}"
        return RewardInfo(1.0 if success else 0.0, success, grounded=True, reason=reason)

    # -- REAL official agent (B0 / sampling) ---------------------------- #
    def run_agent_rollout(self, config, seed: int):
        """Run the **official OfficeBench agent** and return ``(Trajectory, RewardInfo)``.

        The agent emits office-app actions executed in-container through the official ``apps`` modules,
        and the official ``utils/evaluate.py`` checks score the resulting ``/testbed`` — so a single
        rollout reproduces the official OfficeBench number.
        """
        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment
        from .officebench_agent import run_officebench_agent

        solver = config.solver
        model = solver.llm_agent or azure_deployment()
        chat = make_chat_client(model=model, seed=seed)
        max_steps = solver.max_steps if solver.max_steps and solver.max_steps <= 40 else _scaled_max_steps(self.level)

        session = self._new_session(f"s{seed}").start()
        try:
            run = run_officebench_agent(session, self._config, chat, max_steps=max_steps,
                                        temperature=solver.temperature)
            traj = self._trajectory_from_run(run)
            ri = self._score(session)
        finally:
            session.close()
        return traj, ri

    def run_agent_rollout_wm(self, config, method: str, seed: int, budget, cache=None):
        """Official agent interacting with the **world model** (no real writes), scored by the verifier.

        The official agent runs, but tool execution is routed through
        :class:`~tts.env.officebench_wm_predict.OfficeBenchWMPredictor` (the τ²/AppWorld contract):
        **read** actions execute exactly against the pristine ``/testbed`` (a read of a file a prior
        write changed is WM-predicted), while every **write is LLM-predicted and NEVER executed**, so
        the shadow environment is untouched.  The success verifier then judges the **predicted** final
        state (predicted file contents + the write-log), not real files — because in this mode the real
        files are still the initial ones.  The realized actions ride on the trajectory so the selected
        one can be committed (really executed) and officially scored.
        Returns ``(Trajectory, predicted RewardInfo, None)``.
        """
        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment
        from .officebench_agent import run_officebench_agent
        from .officebench_wm import (
            OfficeBenchVerifier,
            reference_files_from_config,
            state_files_from_config,
            state_grounding_enabled,
        )
        from .officebench_wm import _reference_enabled
        from .officebench_wm_predict import OfficeBenchWMPredictor

        solver = config.solver
        model = solver.llm_agent or azure_deployment()
        chat = make_chat_client(model=model, seed=seed)
        predictor = OfficeBenchWMPredictor(make_chat_client(model=model))
        verifier = OfficeBenchVerifier(make_chat_client(model=model))
        max_steps = solver.max_steps if solver.max_steps and solver.max_steps <= 40 else _scaled_max_steps(self.level)

        # read_only: the sampling container is used for EXACT READS ONLY; a write action would raise.
        session = self._new_session(f"wm{seed}")
        session.read_only = True
        session.start()
        try:
            # Write-safety anchor: the shadow /testbed must be byte-for-byte identical after the whole
            # rollout — sampling AND judging (the verifier's state/reference reads) — so that no write
            # (predicted or incidental) ever mutates the environment during selection.
            pristine_before = session.testbed_digest() if _assert_write_safe() else None
            run = run_officebench_agent(session, self._config, chat, max_steps=max_steps,
                                        temperature=solver.temperature, predictor=predictor)
            traj = self._trajectory_from_run(run)
            # Reference-ground on OfficeBench's native ground truth: the gold expected-output files
            # (exact_match tasks) are read-only starting artifacts, so they are read exactly.  Resolved
            # BEFORE the state so a gold-backed file is judged from initial+write-log vs the gold (no
            # fabricated reconstruction), rather than an LLM-reconstructed final file.
            gold_files = None
            if _reference_enabled():
                ref_specs = reference_files_from_config(self._config)
                if ref_specs:
                    try:
                        raw = session.read_reference(ref_specs)
                        spec_by_path = {s["path"]: s.get("label", s["path"]) for s in ref_specs}
                        gold_files = {spec_by_path.get(p, p): c for p, c in raw.items()}
                    except Exception:  # noqa: BLE001
                        gold_files = None
            gold_labels = {os.path.basename(str(k)) for k in (gold_files or {})}
            # Ground the verifier on the PREDICTED resulting state (never the real files, which the WM
            # left pristine).  A file that has a gold reference is shown as initial-content + write-log
            # (the verifier compares that delta against the gold); only files WITHOUT a gold reference
            # are LLM-reconstructed (the trustworthy-as-possible fallback for value/keyword checks).
            final_state = None
            if state_grounding_enabled():
                try:
                    final_state = self._predicted_state(
                        session, predictor, state_files_from_config(self._config), gold_labels)
                except Exception:  # noqa: BLE001 - fall back to the write-log-only judge on any error
                    final_state = {"WRITE LOG (predicted)": predictor.predicted_write_log()}
            confidence = verifier.predict_success(
                self._instruction, run.actions, run.observations, config=self._config,
                final_state=final_state, final_answer=run.final_answer, gold_files=gold_files)
            if pristine_before is not None:
                pristine_after = session.testbed_digest()
                if pristine_after != pristine_before:
                    raise RuntimeError(
                        "OfficeBench world-model write-safety violation: the shadow /testbed was mutated "
                        "during sampling/judging (no write must reach the environment before commit)")
        finally:
            session.close()
        predicted = RewardInfo(reward=float(confidence), success=confidence >= 0.5, grounded=False,
                               reason=(f"OfficeBench WM verifier confidence {confidence:.2f} "
                                       f"({predictor.n_writes} writes predicted, "
                                       f"{predictor.n_failed_writes} predicted-failed, "
                                       f"{predictor.n_dirty_reads} dirty reads predicted, "
                                       f"{predictor.n_reads} reads exact)"))
        return traj, predicted, None

    def _predicted_state(self, session, predictor, files, gold_labels=None) -> dict:
        """Predicted final content of each checked file, for the success verifier.

        A file the trajectory never wrote keeps its exact initial content (read read-only).  A written
        file WITHOUT a gold reference is WM-reconstructed (initial + write-log), the conservative
        fallback for value/keyword checks.  A written file WITH a gold reference (``gold_labels``) is
        NOT reconstructed — it is surfaced as its exact INITIAL content plus the WRITE actions applied,
        so the verifier reasons about that delta against the gold rather than trusting a fabricated
        final file.  If nothing relevant was written, the predicted write-log is surfaced so the
        verifier still sees what the agent (predictively) did.
        """
        from .officebench_wm_predict import _norm_path

        gold_labels = gold_labels or set()
        # A/B toggle: when set, reproduce the pre-change behaviour (always LLM-reconstruct a written
        # file, even gold-backed ones) instead of the initial+write-log delta view.
        reconstruct_gold = os.environ.get("TTS_OFFICEBENCH_WM_RECONSTRUCT_GOLD", "0") in ("1", "true", "True")
        out = {}
        initial = {}
        if files:
            try:
                initial = session.read_state(files)  # exact starting content (container is pristine)
            except Exception:  # noqa: BLE001
                initial = {}
        for spec in files or []:
            path = spec.get("path", "")
            key = path
            init = initial.get(path, "[file does not exist]")
            if not predictor.applied_writes.get(_norm_path(path)):
                out[key] = init  # untouched by the trajectory
            elif os.path.basename(_norm_path(path)) in gold_labels and not reconstruct_gold:
                # gold-backed: judge from initial + write-log vs the gold (no reconstruction).
                out[key] = (f"INITIAL content (before the trajectory):\n{init}\n\n"
                            f"WRITE actions the agent applied to this file (predicted, not executed):\n"
                            f"{predictor.file_write_log(path)}")
            else:
                out[key] = predictor.predict_file_state(path, init)
        # always include the write-log + the agent's answer so give-ups/other-file writes are visible
        out["WRITE LOG (predicted)"] = predictor.predicted_write_log()
        return out

    # -- stepping / evaluation / commit --------------------------------- #
    def step(self, action: Action) -> Observation:  # pragma: no cover - agent loop runs in-session
        raise NotImplementedError("OfficeBench steps run inside the official agent loop")

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        """Replay a trajectory in a fresh container and score it with the official checks."""
        session = self._new_session("eval").start()
        try:
            self._replay(session, trajectory)
            return self._score(session)
        finally:
            session.close()

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        ri = self.evaluate(trajectory)  # single real write (replay the committed actions)
        self._commits += 1
        return ri

    def state_hash(self) -> str:
        return f"officebench:{self.task_id}"  # no shared mutable state across rollouts

    def committed_count(self) -> int:
        return self._commits

    def close(self) -> None:  # pragma: no cover - nothing persistent is held open
        return None


__all__ = ["OfficeBenchTaskEnv", "MUTATING_TOOLS", "real_officebench_available"]
