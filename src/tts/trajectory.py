"""Trajectory-level data structures.

The unit of test-time scaling here is a **trajectory**: an ordered sequence of agent
steps (each a real tool-call / code block executed through the *official* environment
interface) ending in a terminal evaluation.  ``N`` parallel trajectories are sampled, the
single most-promising one is selected, and **its whole action sequence is committed** to the
one real environment.

These types are env-agnostic: ``Action`` (already defined in :mod:`tts.types`) names a tool
and its arguments; an :class:`Observation` is whatever the environment returns for a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import Action


@dataclass
class Observation:
    """The environment's response to one executed step (tool result / code stdout)."""

    content: str = ""
    error: bool = False
    raw: Any = None


@dataclass
class Step:
    """One agent step: a proposed :class:`Action` and the resulting :class:`Observation`."""

    action: Action
    observation: Optional[Observation] = None


@dataclass
class Trajectory:
    """An ordered sequence of steps produced by an agent for a single task."""

    steps: List[Step] = field(default_factory=list)
    final_message: str = ""
    #: terminal flag set once the agent decides it is done (cohering to AGENT_STOP/USER_STOP).
    done: bool = True

    @property
    def actions(self) -> List[Action]:
        return [s.action for s in self.steps]

    def signature(self) -> str:
        return " | ".join(str(a) for a in self.actions)

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class RewardInfo:
    """A reward verdict for a trajectory (predicted by a world model, or the real outcome)."""

    reward: float = 0.0
    success: bool = False
    #: ``True`` for a real/official evaluation, ``False`` for a world-model prediction.
    grounded: bool = False
    reason: str = ""


@dataclass
class RolloutResult:
    """One sampled trajectory plus the world-model's (or oracle's) verdict on it."""

    index: int
    trajectory: Trajectory
    predicted: RewardInfo
    #: realized, grounded official reward for this rollout (real-agent path); ``None`` otherwise.
    grounded: Optional[RewardInfo] = None
    #: recursion telemetry (SpawnNode root; ``None`` for B0 / un-simulated rollouts).
    spawn_root: Optional["object"] = None
    n_reads: int = 0
    n_spawns: int = 0
    depth: int = 0
    tokens: int = 0
    cost: float = 0.0
    score: float = 0.0


__all__ = ["Observation", "Step", "Trajectory", "RewardInfo", "RolloutResult"]
