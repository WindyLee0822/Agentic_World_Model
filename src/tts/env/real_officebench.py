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

from typing import Dict, List, Optional, Sequence

import json
import os
import re

from ..abcs import EntityStore, RestrictedView, TaskEnvironment, WorldModel
from ..config import SimulatorConfig
from ..trajectory import Observation, RewardInfo, RolloutResult, Step, Trajectory
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


def _parse_comparative_best(response: str, n_candidates: int) -> int:
    match = re.search(r"\bBEST\s*:\s*(\d+)\b", response or "", flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"OfficeBench comparative verifier omitted BEST index: {response!r}")
    best = int(match.group(1))
    if not 0 <= best < n_candidates:
        raise ValueError(
            f"OfficeBench comparative verifier returned invalid BEST index {best} "
            f"for {n_candidates} candidates"
        )
    return best


def _scaled_max_steps(level: int) -> int:
    """Return the fixed OfficeBench productive-step budget."""
    del level
    return min(
        _int_env("TTS_OFFICEBENCH_STEPS_CAP", 40),
        _int_env("TTS_OFFICEBENCH_STEPS_FLOOR", 40),
    )


def _rollout_max_steps(configured: int, level: int) -> int:
    fixed = _scaled_max_steps(level)
    return min(configured, fixed) if configured else fixed


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

    def refine_rollout_scores(
        self,
        config,
        rollouts: Sequence[RolloutResult],
    ) -> None:
        """Comparatively audit ambiguous additions while preserving the lower-N winner."""
        enabled = os.environ.get(
            "TTS_OFFICEBENCH_COMPARATIVE_RERANK",
            os.environ.get("TTS_OFFICEBENCH_COMPARATIVE_ZERO_RERANK", "0"),
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled or len(rollouts) < 2:
            return

        try:
            selection_margin = max(
                0.0,
                float(os.environ.get(
                    "TTS_OFFICEBENCH_SELECTION_CONFIDENCE_MARGIN", "0.05")),
            )
        except (TypeError, ValueError):
            selection_margin = 0.05
        try:
            audit_margin = max(
                selection_margin,
                float(os.environ.get(
                    "TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN", "0.25")),
            )
        except (TypeError, ValueError):
            audit_margin = 0.25

        previous = []
        for rollout in rollouts:
            record = (rollout.predicted.metadata or {}).get("comparative_rerank")
            if (
                isinstance(record, dict)
                and 1 < int(record.get("pool_size", 0)) < len(rollouts)
            ):
                previous.append(record)
        prior = max(previous, key=lambda item: int(item["pool_size"])) if previous else None
        prior_pool_size = int(prior["pool_size"]) if prior else 1
        incumbent = int(prior["best"]) if prior else 0
        if not 0 <= incumbent < prior_pool_size:
            raise ValueError(
                f"invalid prior OfficeBench comparative winner {incumbent} "
                f"for pool size {prior_pool_size}"
            )

        def priority(index: int) -> tuple[bool, bool]:
            rollout = rollouts[index]
            metadata = rollout.predicted.metadata if rollout.predicted is not None else {}
            return (
                bool(rollout.trajectory.done),
                not bool((metadata or {}).get("hard_invalid", False)),
            )

        def compare_two(candidate_a: int, candidate_b: int) -> tuple[int, str]:
            pair = [candidate_a, candidate_b]
            rendered = []

            for index in pair:
                rollout = rollouts[index]
                metadata = rollout.predicted.metadata or {}
                state = metadata.get("verifier_final_state") or {}
                audit = metadata.get("verifier_percheck") or {}

                rendered.append(
                    f"### Candidate {index}\n"
                    f"confidence={float(rollout.predicted.reward):.4f}; "
                    f"done={rollout.trajectory.done}; "
                    f"hard_invalid={bool(metadata.get('hard_invalid', False))}; "
                    f"termination={rollout.trajectory.termination_reason!r}\n"
                    f"per-check: "
                    f"met={audit.get('met')}, "
                    f"unmet={audit.get('unmet')}, "
                    f"unknown={audit.get('unknown')}, "
                    f"total={audit.get('total')}\n"
                    f"predicted final state:\n"
                    f"{_compact_final_state(state)}\n"
                    f"trajectory diagnostics:\n"
                    f"{_compact_trajectory(rollout.trajectory)}"
                )

            prompt = (
                "Choose the candidate most likely to pass every native "
                "OfficeBench evaluator check. The predicted final state is "
                "the primary evidence. Use trajectory diagnostics only to "
                "detect failed, malformed, incomplete, or destructive actions.\n\n"
                f"## User task\n{self._instruction}\n\n"
                f"## Native evaluator checks\n"
                f"{json.dumps(self._config.get('evaluation', []), default=str)[:12000]}\n\n"
                f"{chr(10).join(rendered)}\n\n"
                f"Valid candidate IDs: {pair}\n"
                "Return exactly:\n"
                "BEST: <candidate ID>\n"
                "REASON: <one concise sentence>"
            )

            response = client.chat(
                "You are a precise comparative OfficeBench trajectory verifier.",
                prompt,
                temperature=0.0,
                max_tokens=256,
                extra_body=extra_body,
            )

            return _parse_comparative_best(response, len(rollouts)), response

        new_indices = list(range(prior_pool_size, len(rollouts)))
        if not new_indices:
            return
        # A predicted task failure is not proof that its completed trajectory is invalid.
        # Keep it eligible in a zero-confidence pool, even if a precheck rejected its state.
        zero_pool = all(float(r.predicted.reward) == 0.0 for r in rollouts)
        if zero_pool:
            def priority(index):
                return (bool(rollouts[index].trajectory.done), True)
        challenger = max(
            new_indices,
            key=lambda index: (
                priority(index),
                float(rollouts[index].predicted.reward),
                -float(rollouts[index].cost),
                -index,
            ),
        )
        incumbent_priority = priority(incumbent)
        challenger_priority = priority(challenger)
        incumbent_score = float(rollouts[incumbent].predicted.reward)
        challenger_score = float(rollouts[challenger].predicted.reward)
        native_best = incumbent
        if (
            challenger_priority > incumbent_priority
            or (
                challenger_priority == incumbent_priority
                and challenger_score > incumbent_score + selection_margin
            )
        ):
            native_best = challenger

        native_priority = priority(native_best)
        native_score = float(rollouts[native_best].predicted.reward)
        contenders = [
            index
            for index in [incumbent, *new_indices]
            if priority(index) == native_priority
            and (
                index == incumbent
                or float(rollouts[index].predicted.reward) >= native_score - audit_margin
            )
        ]
        contenders = list(dict.fromkeys(contenders))
        best = native_best
        response = None

        from ..llm import make_chat_client
        from ..llm.azure import azure_deployment

        # if len(contenders) >= 2:
        #     rendered = []
        #     for index in contenders:
        #         rollout = rollouts[index]
        #         steps = "\n".join(
        #             f"  {step_index}. action={_action_of(step.action)}\n"
        #             f"     observation={str(step.observation.content)[:500]}"
        #             for step_index, step in enumerate(rollout.trajectory.steps)
        #         )
        #         metadata = rollout.predicted.metadata or {}
        #         state = metadata.get("verifier_final_state") or {}
        #         rendered.append(
        #             f"### Candidate {index}\n"
        #             f"confidence={float(rollout.predicted.reward):.4f}; "
        #             f"done={rollout.trajectory.done}; "
        #             f"hard_invalid={bool(metadata.get('hard_invalid', False))}; "
        #             f"termination={rollout.trajectory.termination_reason!r}\n"
        #             f"per-check audit={json.dumps(metadata.get('verifier_percheck'), default=str)}\n"
        #             f"predicted final state={json.dumps(state, default=str)[:12000]}\n"
        #             f"trajectory:\n{steps or '  (no actions)'}"
        #         )
        #     prompt = (
        #         "Choose the ONE candidate most likely to pass every native OfficeBench evaluator check. "
        #         "Audit the predicted final state against every check, then use the action/observation "
        #         "trace to detect malformed, failed, incomplete, or destructive edits. Confidence values "
        #         "are noisy and tied candidates are common, so do not prefer the incumbent or a higher "
        #         "index without evaluator-visible evidence. If all candidates are flawed, choose the "
        #         "least flawed.\n\n"
        #         f"## User task\n{self._instruction}\n\n"
        #         f"## Native evaluator checks\n"
        #         f"{json.dumps(self._config.get('evaluation', []), default=str)[:12000]}\n\n"
        #         f"{chr(10).join(rendered)}\n\n"
        #         f"Valid candidate IDs: {contenders}\n"
        #         "Return exactly:\nBEST: <candidate ID>\nREASON: <one concise sentence>"
        #     )
        #     model = config.solver.llm_agent or azure_deployment()
            
        #     client = make_chat_client(
        #         model=model,
        #         seed=20_000_000 + int(config.seed) + len(rollouts),
        #     )
        #     extra_body = None
        #     if (
        #         getattr(client, "backend", "") == "openai"
        #         and "qwen" in str(model).lower()
        #     ):
        #         extra_body = {
        #             "chat_template_kwargs": {
        #                 "enable_thinking": False,
        #             }
        #         }

        #     response = client.chat(
        #         "You are a precise comparative OfficeBench trajectory verifier.",
        #         prompt,
        #         temperature=0.0,
        #         max_tokens=256,
        #         extra_body=extra_body,
        #     )
        #     best = _parse_comparative_best(response, len(rollouts))
        #     if best not in contenders:
        #         raise ValueError(
        #             f"OfficeBench comparative verifier selected non-contender {best}; "
        #             f"valid contenders were {contenders}"
        #         )
        zero_confidence_pool = all(float(rollouts[i].predicted.reward) == 0.0 for i in contenders)
        if zero_confidence_pool:
            # A free-text comparison cannot manufacture evidence for a zero-score
            # replacement. Use the explicit per-check audit, retaining the incumbent
            # on ties or incomplete audits. This does not turn partial credit into success.
            def audited_progress(index):
                audit = (rollouts[index].predicted.metadata or {}).get('verifier_percheck') or {}
                total = audit.get('total', 0)
                if not total or audit.get('unknown', 0) or not audit.get('met'):
                    return -1.0
                return float(audit['met']) / float(total)
            best = native_best
            for index in contenders:
                if audited_progress(index) > audited_progress(best):
                    best = index
            response = 'Zero-confidence guard: replace only for a strictly better complete per-check audit; retain incumbent on ties.'
        if len(contenders) >= 2 and not zero_confidence_pool:
            model = config.solver.llm_agent or azure_deployment()
            is_qwen = "qwen" in str(model).lower()

            # ================================================================
            # GPT-5.4: KEEP THE ORIGINAL LISTWISE COMPARATIVE VERIFIER EXACTLY
            # ================================================================
            if not is_qwen:
                rendered = []
                for index in contenders:
                    rollout = rollouts[index]
                    steps = "\n".join(
                        f"  {step_index}. action={_action_of(step.action)}\n"
                        f"     observation={str(step.observation.content)[:500]}"
                        for step_index, step in enumerate(rollout.trajectory.steps)
                    )
                    metadata = rollout.predicted.metadata or {}
                    state = metadata.get("verifier_final_state") or {}
                    rendered.append(
                        f"### Candidate {index}\n"
                        f"confidence={float(rollout.predicted.reward):.4f}; "
                        f"done={rollout.trajectory.done}; "
                        f"hard_invalid={bool(metadata.get('hard_invalid', False))}; "
                        f"termination={rollout.trajectory.termination_reason!r}\n"
                        f"per-check audit={json.dumps(metadata.get('verifier_percheck'), default=str)}\n"
                        f"predicted final state={json.dumps(state, default=str)[:12000]}\n"
                        f"trajectory:\n{steps or '  (no actions)'}"
                    )

                prompt = (
                    "Choose the ONE candidate most likely to pass every native OfficeBench evaluator check. "
                    "Audit the predicted final state against every check, then use the action/observation "
                    "trace to detect malformed, failed, incomplete, or destructive edits. Confidence values "
                    "are noisy and tied candidates are common, so do not prefer the incumbent or a higher "
                    "index without evaluator-visible evidence. If all candidates are flawed, choose the "
                    "least flawed.\n\n"
                    f"## User task\n{self._instruction}\n\n"
                    f"## Native evaluator checks\n"
                    f"{json.dumps(self._config.get('evaluation', []), default=str)[:12000]}\n\n"
                    f"{chr(10).join(rendered)}\n\n"
                    f"Valid candidate IDs: {contenders}\n"
                    "Return exactly:\nBEST: <candidate ID>\nREASON: <one concise sentence>"
                )

                response = make_chat_client(
                    model=model,
                    seed=20_000_000 + int(config.seed) + len(rollouts),
                ).chat(
                    "You are a precise comparative OfficeBench trajectory verifier.",
                    prompt,
                    temperature=0.0,
                    max_tokens=256,
                )

                best = _parse_comparative_best(
                    response,
                    len(rollouts),
                )

                if best not in contenders:
                    raise ValueError(
                        f"OfficeBench comparative verifier selected non-contender {best}; "
                        f"valid contenders were {contenders}"
                    )

            # ================================================================
            # QWEN: PAIRWISE COMPARISON TO AVOID HUGE N-WAY PROMPTS
            # ================================================================
            else:
                client = make_chat_client(
                    model=model,
                    seed=20_000_000 + int(config.seed) + len(rollouts),
                )

                # Keep your Qwen verifier-thinking fix here.
                extra_body = {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                }

                def render_qwen_candidate(index: int) -> str:
                    rollout = rollouts[index]
                    metadata = rollout.predicted.metadata or {}
                    state = metadata.get("verifier_final_state") or {}

                    # Qwen-specific compact trace:
                    # preserve the full action sequence, but do not dump
                    # 500 chars of observation for every step.
                    steps = []
                    for step_index, step in enumerate(rollout.trajectory.steps):
                        action_text = str(_action_of(step.action))
                        observation_text = str(step.observation.content or "")

                        # Actions are much more informative than full observations
                        # for the comparative audit.
                        steps.append(
                            f"  {step_index}. "
                            f"action={action_text[:400]}\n"
                            f"     observation={observation_text[:120]}"
                        )

                    return (
                        f"### Candidate {index}\n"
                        f"confidence={float(rollout.predicted.reward):.4f}; "
                        f"done={rollout.trajectory.done}; "
                        f"hard_invalid={bool(metadata.get('hard_invalid', False))}; "
                        f"termination={rollout.trajectory.termination_reason!r}\n"
                        f"per-check audit="
                        f"{json.dumps(metadata.get('verifier_percheck'), default=str)}\n"
                        f"predicted final state="
                        f"{json.dumps(state, default=str)[:8000]}\n"
                        f"trajectory:\n"
                        f"{chr(10).join(steps) or '  (no actions)'}"
                    )

                # Start from the native winner, then challenge it one candidate
                # at a time. At most TWO trajectories enter any Qwen prompt.
                best = native_best
                pairwise_responses = []

                for challenger_index in contenders:
                    if challenger_index == best:
                        continue

                    pair = [best, challenger_index]

                    rendered = [
                        render_qwen_candidate(index)
                        for index in pair
                    ]

                    prompt = (
                        "Choose the ONE candidate most likely to pass every native "
                        "OfficeBench evaluator check. Audit the predicted final state "
                        "against every check, then use the action/observation trace "
                        "to detect malformed, failed, incomplete, or destructive edits. "
                        "Confidence values are noisy, so use evaluator-visible evidence. "
                        "If both candidates are flawed, choose the least flawed.\n\n"
                        f"## User task\n{self._instruction}\n\n"
                        f"## Native evaluator checks\n"
                        f"{json.dumps(self._config.get('evaluation', []), default=str)[:12000]}\n\n"
                        f"{chr(10).join(rendered)}\n\n"
                        f"Valid candidate IDs: {pair}\n"
                        "Return exactly:\n"
                        "BEST: <candidate ID>\n"
                        "REASON: <one concise sentence>"
                    )

                    pair_response = client.chat(
                        "You are a precise comparative OfficeBench trajectory verifier.",
                        prompt,
                        temperature=0.0,
                        max_tokens=256,
                        extra_body=extra_body,
                    )

                    pair_best = _parse_comparative_best(
                        pair_response,
                        len(rollouts),
                    )

                    if pair_best not in pair:
                        raise ValueError(
                            f"OfficeBench pairwise verifier selected "
                            f"non-contender {pair_best}; "
                            f"valid candidates were {pair}"
                        )

                    best = pair_best

                    pairwise_responses.append(
                        f"PAIR {pair}: {pair_response}"
                    )

                # Keep metadata structure compatible with the old code:
                # response remains one serializable string.
                response = "\n\n".join(pairwise_responses)

                if best not in contenders:
                    raise ValueError(
                        f"OfficeBench comparative verifier selected non-contender {best}; "
                        f"valid contenders were {contenders}"
                    )
        decision = {
            "pool_size": len(rollouts),
            "prior_pool_size": prior_pool_size,
            "incumbent": incumbent,
            "native_best": native_best,
            "contenders": contenders,
            "best": best,
            "response": response,
        }
        override = {
            "pool_size": len(rollouts),
            "best": best,
            "source": "officebench_comparative",
        }
        for rollout in rollouts:
            rollout.predicted.metadata["comparative_rerank"] = decision
            rollout.predicted.metadata["selection_override"] = override

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
                observation=Observation(content=str(obs),
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
            if act == "switch_app" or act == "got_stuck" or not action or "app" not in action:
                continue  # system bookkeeping: no container effect
            if act == "finish_task":
                session.write_answer(str(action.get("answer", "None")))
                continue
            result = session.exec_action(action, workdir)
            if os.environ.get("TTS_OFFICEBENCH_COMMIT_TRACE", "0") == "1":
                print("[officebench-commit-trace] " + json.dumps({
                    "task_id": self.task_id, "action": action,
                    "predicted": step.observation.content,
                    "actual": str(result.get("observation", result)),
                }), flush=True)
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
        max_steps = _rollout_max_steps(solver.max_steps, self.level)

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
        reads execute exactly only in the explicit clean-read ablation. Under
        ``predict_all_actions``, recursive Ours may read immutable objects through the driver but cannot
        dispatch any OfficeBench app or shell action; flat B1 predicts from fixed context. Every
        candidate read/write observation is generated by restricted Python, so the shadow environment
        is untouched. The success verifier then judges the **predicted** final
        state (predicted file contents + the write-log), not real files — because in this mode the real
        files are still the initial ones.  The realized actions ride on the trajectory so the selected
        one can be committed (really executed) and officially scored.

        Under B1, the app catalogue, evaluator criteria, initial checked files, reference files, and
        testbed listing are frozen before the first candidate action and reused by observation and
        success prompts; candidate actions cannot trigger new grounding reads. Under Ours, observation
        and success predictions retain the bounded agentic grounding loop over read-only state.
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
        from .officebench_wm_predict import (
            OfficeBenchWMPredictor,
            OfficeBenchSimulationError,
            build_officebench_flat_system_prompt,
        )
        from .officebench_wm_python import prepare_officebench_python_tool

        solver = config.solver
        model = solver.llm_agent or azure_deployment()
        chat = make_chat_client(model=model, seed=seed)
        predictor = OfficeBenchWMPredictor(make_chat_client(model=model))
        verifier = OfficeBenchVerifier(
            make_chat_client(model=model),
            sample_seed=10_000_000 + seed * 100,
        )
        prepare_officebench_python_tool()
        max_steps = _rollout_max_steps(solver.max_steps, self.level)
        flat_context = str(method).lower() == "b1"

        # read_only: the sampling container is used for EXACT READS ONLY; a write action would raise.
        session = self._new_session(f"wm{seed}")
        session.read_only = True
        session.actions_disabled = bool(config.simulator.predict_all_actions)
        session.start()
        try:
            # Write-safety anchor: the shadow /testbed must be byte-for-byte identical after the whole
            # rollout — sampling AND judging (the verifier's state/reference reads) — so that no write
            # (predicted or incidental) ever mutates the environment during selection.
            pristine_before = session.testbed_digest() if _assert_write_safe() else None
            state_specs = state_files_from_config(self._config)
            ref_specs = (
                reference_files_from_config(self._config)
                if _reference_enabled() else []
            )
            initial_state = None
            gold_files = None
            if flat_context:
                try:
                    initial_state = session.read_state(state_specs) if state_specs else {}
                except Exception:  # noqa: BLE001
                    initial_state = {}
                if ref_specs:
                    try:
                        raw = session.read_reference(ref_specs)
                        spec_by_path = {
                            spec["path"]: spec.get("label", spec["path"])
                            for spec in ref_specs
                        }
                        gold_files = {
                            spec_by_path.get(path, path): content
                            for path, content in raw.items()
                        }
                    except Exception:  # noqa: BLE001
                        gold_files = None
                testbed_listing = ""
                if not config.simulator.predict_all_actions:
                    try:
                        listing_result = session.exec_action(
                            {
                                "app": "shell",
                                "action": "command",
                                "command": (
                                    "find /testbed/data -maxdepth 3 -type f "
                                    "2>/dev/null | sort"
                                ),
                            },
                            "/",
                        )
                        testbed_listing = str(
                            listing_result.get("observation", listing_result))
                    except Exception:  # noqa: BLE001
                        testbed_listing = ""
                flat_system_prompt = build_officebench_flat_system_prompt(
                    task_config=self._config,
                    catalogue=session.apps_catalogue(),
                    initial_state=initial_state,
                    gold_files=gold_files,
                    testbed_listing=testbed_listing,
                    predict_all_actions=config.simulator.predict_all_actions,
                )
                predictor.configure_flat_context(
                    flat_system_prompt, initial_state=initial_state)
                verifier.configure_flat_context(flat_system_prompt)
            run = run_officebench_agent(
                session,
                self._config,
                chat,
                max_steps=max_steps,
                temperature=solver.temperature,
                predictor=predictor,
                predict_all_actions=config.simulator.predict_all_actions,
            )
            traj = self._trajectory_from_run(run)
            # Reference-ground on OfficeBench's native ground truth: the gold expected-output files
            # (exact_match tasks) are read-only starting artifacts, so they are read exactly.  Resolved
            # BEFORE the state so a gold-backed file is judged from initial+write-log vs the gold (no
            # fabricated reconstruction), rather than an LLM-reconstructed final file.
            if not flat_context and _reference_enabled():
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
                        session,
                        predictor,
                        state_specs,
                        gold_labels,
                        initial_state=initial_state,
                    )
                except OfficeBenchSimulationError:
                    raise  # unknown state must reach the rollout retry/error path
                except Exception:  # noqa: BLE001 - fall back to the write-log-only judge on any error
                    final_state = {"WRITE LOG (predicted)": predictor.predicted_write_log()}
            verifier.bind_query_interface(predictor.query_interface)
            confidence = verifier.predict_success(
                self._instruction, run.actions, run.observations, config=self._config,
                final_state=final_state, final_answer=run.final_answer, gold_files=gold_files)
            if pristine_before is not None:
                pristine_after = session.testbed_digest()
                if pristine_after != pristine_before:
                    raise RuntimeError(
                        "OfficeBench world-model write-safety violation: the shadow /testbed was mutated "
                        "during sampling/judging (no write must reach the environment before commit)")
        except OfficeBenchSimulationError as exc:
            exc.simulation_metrics = {
                "python_calls": predictor.n_python_calls + verifier.n_python_calls,
                "python_failures": predictor.n_python_failures + verifier.n_python_failures,
                "python_forced": predictor.n_python_forced + verifier.n_python_forced,
                "python_mode_misses": predictor.n_python_mode_misses + verifier.n_python_mode_misses,
            }
            raise
        finally:
            session.close()
        python_calls = predictor.n_python_calls + verifier.n_python_calls
        python_failures = predictor.n_python_failures + verifier.n_python_failures
        python_forced = predictor.n_python_forced + verifier.n_python_forced
        python_mode_misses = (
            predictor.n_python_mode_misses + verifier.n_python_mode_misses)
        predicted = RewardInfo(reward=float(confidence), success=confidence >= 0.5, grounded=False,
                               reason=(f"OfficeBench WM verifier confidence {confidence:.2f} "
                                       f"({predictor.n_writes} writes predicted, "
                                       f"{predictor.n_failed_writes} predicted-failed, "
                                       f"{predictor.n_predicted_reads} reads predicted, "
                                       f"{predictor.n_dirty_reads} dirty reads predicted, "
                                       f"{predictor.n_reads} reads exact, "
                                       f"{python_calls} Python calls)"
                                       + (f"; deterministic failure: {verifier.last_precheck_failure}"
                                          if verifier.last_precheck_failure else "")),
                               metadata={
                                   "python_calls": python_calls,
                                   "python_failures": python_failures,
                                   "python_forced": python_forced,
                                   "python_mode_misses": python_mode_misses,
                                   "verifier_percheck": verifier.last_percheck_breakdown,
                                   "partial_progress": (verifier.last_percheck_breakdown or {}).get("atomic_score", 0.0),
                                   "verifier_final_state": final_state,
                                   "hard_invalid": verifier.last_precheck_failure is not None,
                               })
        return traj, predicted, None

    def _predicted_state(
        self,
        session,
        predictor,
        files,
        gold_labels=None,
        *,
        initial_state=None,
    ) -> dict:
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
        initial = dict(initial_state or {})
        if files and initial_state is None:
            try:
                initial = session.read_state(files)  # exact starting content (container is pristine)
            except Exception:  # noqa: BLE001
                initial = {}
        for spec in files or []:
            path = spec.get("path", "")
            key = path
            init = initial.get(path, "[file does not exist]")
            target = _norm_path(path)
            if spec.get('doc_type') == 'email':
                out[key] = predictor.predict_file_state(path, init)
            elif not predictor.was_written(target):
                out[key] = init  # untouched by the trajectory
            elif ((spec.get("doc_type", "txt") in ("txt", "pdf") or target.lower().endswith('.pdf'))
                  and predictor.applied_writes.get(target)
                  and predictor.applied_writes[target][-1][0].get('app') == 'word'
                  and predictor.applied_writes[target][-1][0].get('action') in ('create_new_file', 'write_to_file')):
                out[key] = "[incompatible file format: Word produced DOCX; required output is plain text or PDF]"
            elif predictor.is_deleted(target):
                out[key] = predictor.predict_file_state(path, init)
            elif os.path.basename(target) in gold_labels and not reconstruct_gold:
                # gold-backed: judge from initial + write-log vs the gold (no reconstruction).
                exact = predictor._exact_overlay_content(target, init)
                if exact is not None:
                    out[key] = exact
                    continue
                out[key] = (f"INITIAL content (before the trajectory):\n{init}\n\n"
                            f"WRITE actions the agent applied to this file (predicted, not executed):\n"
                            f"{predictor.file_write_log(path)}")
            elif predictor.global_dirty:
                out[key] = predictor.predict_file_state(path, init)
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
