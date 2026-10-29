"""Solver agents that propose trajectories through a stepper.

Agents see only a :class:`~tts.abcs.RestrictedView` (task instruction + tool names) and never
touch the real environment; a driver executes their proposed actions against a stepper (the
real env for the committed rollout, or a world model while sampling).

* :class:`ScriptedAgent` — a deterministic, LLM-free stand-in for a sampling policy.  Given a
  *plan* (the reference action sequence) and a per-rollout seed, it either follows the plan
  (probability ``p_correct``) or emits a perturbed, failing variant.  This is what makes the
  whole pipeline runnable and testable without any LLM calls, while still producing the diverse
  pool of good/bad trajectories that best-of-``N`` selection ranks.
* :class:`LLMAgent` — a minimal adapter for an interactive, observation-driven solver.  The
  **real official-prompt agents** that produce real rollouts (and the sampled rollouts) are not driven
  through this class; they live in the env wrappers (``Tau2TaskEnv.run_agent_rollout`` runs the
  official τ² orchestrator; ``AppWorldTaskEnv.run_agent_rollout`` runs the official ReAct loop),
  so the official prompts and evaluators are used verbatim.  This adapter remains for plugging a
  custom step-wise decision function into the same harness.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .abcs import Agent, RestrictedView
from .trajectory import Trajectory
from .types import Action


class ScriptedAgent(Agent):
    """Deterministic stand-in for a sampling solver.

    ``plan`` is the reference (gold) action sequence — the agent's *hypothesis*, not a
    privileged DB read.  With probability ``p_correct`` a rollout follows the plan exactly;
    otherwise it perturbs the plan into a failing variant (dropping one mutating action), giving
    the rollout pool the good/bad mix that selection must discriminate.
    """

    name = "scripted"

    def __init__(self, plan: List[Action], mutating_names: set, p_correct: float = 0.5) -> None:
        self._plan = list(plan)
        self._mutating = set(mutating_names)
        self._p = p_correct
        self._queue: List[Action] = []

    def reset(self, view: RestrictedView, seed: int) -> None:
        rng = random.Random((seed + 1) * 100003)
        if not self._plan or rng.random() < self._p:
            self._queue = list(self._plan)
        else:
            self._queue = self._perturb(self._plan, rng)

    def _perturb(self, plan: List[Action], rng: random.Random) -> List[Action]:
        mut_idx = [i for i, a in enumerate(plan) if a.name in self._mutating]
        if mut_idx:
            drop = rng.choice(mut_idx)  # drop a required mutating step -> fails official eval
            return [a for i, a in enumerate(plan) if i != drop]
        return plan[:-1] if len(plan) > 1 else []  # no mutation: drop the last step

    def next_action(self, history: Trajectory) -> Optional[Action]:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def final_message(self, history: Trajectory) -> str:
        return "Done."


class LLMAgent(Agent):
    """Adapter for an interactive LLM agent (decides the next action from observations).

    Kept minimal and optional: ``decide`` is injected so the harness can plug the official
    benchmark agent or an :class:`~tts.llm` client without this module importing a concrete LLM.
    """

    name = "llm"

    def __init__(self, decide, max_steps: int = 12) -> None:
        self._decide = decide  # callable(view, history) -> Optional[Action]
        self._max_steps = max_steps
        self._view: Optional[RestrictedView] = None

    def reset(self, view: RestrictedView, seed: int) -> None:
        self._view = view
        self._seed = seed

    def next_action(self, history: Trajectory) -> Optional[Action]:
        if len(history) >= self._max_steps:
            return None
        return self._decide(self._view, history)


__all__ = ["ScriptedAgent", "LLMAgent"]
