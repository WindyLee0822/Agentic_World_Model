"""Abstract base classes for trajectory-level, read-grounded test-time scaling.

The decoupling rule is unchanged: no component imports a concrete env or LLM class — every
component depends only on these ABCs.  The class hierarchy structurally enforces the
read/write asymmetry and the two subagent roles:

* :class:`EntityStore` is a **privileged but read-only** handle to the real backing database.
  It exposes only reads (returning deep copies), so a world model holding one *cannot* mutate
  the environment.
* :class:`WorldModel` (the *simulator*) predicts the environment's response to an action by
  reading the :class:`EntityStore` (recursively, via probe subagents) and replaying onto a
  private shadow — it never receives the :class:`TaskEnvironment`, so it is structurally
  incapable of committing.
* :class:`Agent` (the *solver*) sees only a :class:`RestrictedView` (task instruction + tool
  schema, **no** privileged reads, **no** commit) and proposes a sequence of actions.
* :class:`TaskEnvironment` is the one real environment and the single writer: only
  :meth:`step` (during the final commit) and :meth:`commit_trajectory` mutate real state.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple

from .config import SimulatorConfig
from .trajectory import Observation, RewardInfo, Trajectory
from .types import Action

# --------------------------------------------------------------------------- #
# Privileged, READ-ONLY backing store (for world models / probe subagents)
# --------------------------------------------------------------------------- #


class EntityStore(abc.ABC):
    """A privileged but **write-incapable** read handle to the real backing database.

    State is exposed as named *entities* (e.g. ``"reservation"``, ``"user"``, ``"flight"`` for
    the τ² airline domain).  Only reads are offered, and reads return deep copies, so the holder
    can never mutate the environment.  Every read is charged to a budget by the caller
    (``node_read_limit`` bounds per-node reads — *no full state in context*).
    """

    @abc.abstractmethod
    def read(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        """Return a deep copy of one entity's row, or ``None`` if absent (read-only)."""

    @abc.abstractmethod
    def list_ids(self, entity_type: str) -> List[str]:
        """List the available ids of an entity type (read-only)."""


# --------------------------------------------------------------------------- #
# Solver-facing restricted view (no privileged reads, no commit)
# --------------------------------------------------------------------------- #


class RestrictedView(abc.ABC):
    """The only interface a solver/agent gets: the task instruction + the tool schema.

    No :class:`EntityStore`, no ``commit``, no ``step`` on the real environment.
    """

    @property
    @abc.abstractmethod
    def task_id(self) -> str: ...

    @abc.abstractmethod
    def instruction(self) -> str:
        """The natural-language task prompt (cohering to the official task instruction)."""

    @abc.abstractmethod
    def tool_schema(self) -> List[str]:
        """Names of the tools the agent may call."""


# --------------------------------------------------------------------------- #
# Agent (solver) — proposes a trajectory through a stepper
# --------------------------------------------------------------------------- #


class Agent(abc.ABC):
    """A solver that proposes actions one step at a time given the history so far.

    The same interface serves a deterministic scripted agent (predetermined plan) and an
    interactive LLM agent (decides the next action from observations).  Agents never touch the
    real environment directly; a driver executes their actions against a *stepper* (the real
    env for the committed rollout, or a :class:`WorldModel` while sampling).
    """

    name: str = "agent"

    @abc.abstractmethod
    def reset(self, view: RestrictedView, seed: int) -> None:
        """Begin a fresh rollout for ``view`` with the given seed."""

    @abc.abstractmethod
    def next_action(self, history: Trajectory) -> Optional[Action]:
        """Return the next action, or ``None`` when the agent is done."""

    def final_message(self, history: Trajectory) -> str:  # pragma: no cover - cosmetic default
        return "Done."


# --------------------------------------------------------------------------- #
# Stepper: anything that can execute an action and return an observation
# --------------------------------------------------------------------------- #


class Stepper(abc.ABC):
    """Executes an action, returning an :class:`Observation`."""

    @abc.abstractmethod
    def step(self, action: Action) -> Observation: ...


# --------------------------------------------------------------------------- #
# World model (the simulator) — read-grounded, never writes the real env
# --------------------------------------------------------------------------- #


class WorldModel(Stepper):
    """Predicts the environment's response to a trajectory without committing.

    Built from an :class:`EntityStore` (read-only) + a :class:`ShadowEvaluator` only — never
    the :class:`TaskEnvironment` — so it is structurally incapable of committing.  ``step``
    predicts one action's observation against a private shadow; :meth:`evaluate` predicts the
    trajectory's terminal reward from read-grounded state.
    """

    name: str = "world_model"

    @abc.abstractmethod
    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        """Predict the trajectory's terminal reward from read-grounded state (no commit)."""

    @property
    @abc.abstractmethod
    def spawn_root(self):
        """The :class:`~tts.simulator.spawn_tree.SpawnNode` recording the read recursion."""


# --------------------------------------------------------------------------- #
# Env-specific dynamics + reward, shared by world models and forks
# --------------------------------------------------------------------------- #


class ShadowEvaluator(abc.ABC):
    """Env-specific read-graph + reward logic that drives the read-grounded world model.

    This is the single place env semantics live; the recursive and flat world models share it
    and differ *only* in how far they read.  It must never write the real env — it operates on
    a *shadow* dict built from privileged reads.

    * :meth:`seed_entities` — the entities a candidate (and the gold reference) start from.
    * :meth:`child_references` — entities a fetched row transitively references (drives recursion).
    * :meth:`reward_from_shadow` — predict the trajectory's reward by replaying it and the gold
      reference onto deep copies of the read-grounded shadow and comparing terminal state.  If
      a needed entity is absent from the shadow (under-reading), the replay cannot be validated
      and the prediction is a miss — which is exactly how the flat world model loses to the
      recursive one.
    """

    @abc.abstractmethod
    def seed_entities(self, candidate_actions: List[Action]) -> List[tuple]:
        """``[(entity_type, entity_id), ...]`` to begin reading from (candidate + gold refs)."""

    @abc.abstractmethod
    def child_references(self, entity_type: str, entity_id: str, row: Dict) -> List[tuple]:
        """Entities transitively referenced by a row (drives recursive reads)."""

    @abc.abstractmethod
    def reward_from_shadow(self, shadow: Dict[tuple, Dict], candidate_actions: List[Action]) -> RewardInfo:
        """Predict the candidate trajectory's reward from the read-grounded shadow (no commit)."""

    def observe(self, shadow: Dict[tuple, Dict], action: Action) -> Optional[Tuple[str, bool]]:
        """Faithful read-grounded observation for one action (optional; default: advisory only).

        When implemented, the env executes the **real tool function** for ``action`` against a
        private scratch built from the read-grounded ``shadow`` (writes isolated to that scratch —
        never the real env) and returns ``(content, is_error)``.  This lets the solver interact with
        the world model and receive genuine tool returns rather than a pseudo presence signal, while
        staying read-grounded and write-safe.  Returning ``None`` (the default) makes the world model
        fall back to the coarse ``ok``/``missing-state`` advisory signal.
        """
        return None


# --------------------------------------------------------------------------- #
# The one real environment (single writer)
# --------------------------------------------------------------------------- #


class TaskEnvironment(Stepper):
    """One real, stateful, non-forkable task environment, cohering to the official runner.

    Hands out a :class:`RestrictedView` to agents and a read-only :class:`EntityStore` to world
    models.  Only :meth:`step` (used during the committed rollout) and
    :meth:`commit_trajectory` mutate real state; :meth:`fork` (a deep snapshot) is permitted
    **only** for the B2 baseline.
    """

    @property
    @abc.abstractmethod
    def env_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def task_id(self) -> str: ...

    @abc.abstractmethod
    def restricted_view(self) -> RestrictedView: ...

    @abc.abstractmethod
    def entity_store(self) -> EntityStore:
        """Privileged, read-only backing-DB handle for world models."""

    @abc.abstractmethod
    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        """Build a read-grounded world model (``config.kind`` = flat | recursive)."""

    @abc.abstractmethod
    def fork(self) -> "TaskEnvironment":
        """A deep snapshot of the real env (B2 baseline ONLY)."""

    @abc.abstractmethod
    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        """Official, grounded evaluation of a trajectory's terminal state."""

    @abc.abstractmethod
    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        """Execute the whole trajectory on the real env (single writer) and official-evaluate."""

    @abc.abstractmethod
    def gold_trajectory(self) -> Trajectory:
        """The reference (gold) action sequence from the official task, for scripted agents."""

    def run_agent_rollout(self, config, seed: int):
        """Run the env's **real, official-prompt agent** once, returning ``(Trajectory, RewardInfo)``.

        The returned :class:`RewardInfo` is the *grounded, official* benchmark reward (so a single
        rollout reproduces the official B0 number).  Real envs override this; the abstract test
        doubles (which exercise the env-agnostic machinery with a scripted stand-in) do not."""
        raise NotImplementedError("this environment has no real official agent wired")

    def run_agent_rollout_wm(self, config, method: str, seed: int, budget, cache=None):
        """Run the **official-prompt agent + user simulator against the WORLD MODEL** once.

        The official orchestrator runs unchanged (faithful prompts + the user simulator dialogue, so
        the agent receives the same kind of information it would on the real env), but every tool
        call is served by the read-grounded world model instead of the real environment: read tools
        are answered exactly (read-only), write tools are WM-predicted, and the real env is never
        mutated.  Returns ``(Trajectory, predicted RewardInfo, spawn_root)`` — the trajectory carries
        the realized official message log so the selected one can be committed/scored faithfully
        (including dual-control user actions).  Only envs with a read-grounded world model override
        this."""
        raise NotImplementedError("this environment has no world-model agent rollout wired")

    @abc.abstractmethod
    def state_hash(self) -> str:
        """A hash of real backing state (to assert the world models never write it)."""

    @abc.abstractmethod
    def committed_count(self) -> int: ...

    # subset/level metadata (optional; for per-subset reporting)
    subset: str = "default"
    split: Optional[str] = None
    level: Optional[int] = None
    domain: Optional[str] = None


__all__ = [
    "EntityStore",
    "RestrictedView",
    "Agent",
    "Stepper",
    "WorldModel",
    "ShadowEvaluator",
    "TaskEnvironment",
]
