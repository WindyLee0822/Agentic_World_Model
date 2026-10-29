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
    #: the realized official message log (τ² ``SimulationRun.messages``) when a trajectory was
    #: produced by the official orchestrator — including BOTH assistant and user (dual-control) tool
    #: calls.  When present, the env commits/scores this faithful log directly (so dual-control
    #: domains score correctly); otherwise it falls back to replaying ``actions`` as assistant calls.
    sim_messages: Optional[List[Any]] = None
    #: the official termination reason for ``sim_messages`` (kept for faithful evaluation).
    termination_reason: Any = None

    @property
    def actions(self) -> List[Action]:
        return [s.action for s in self.steps]

    def signature(self) -> str:
        # A compact identity string for logging/selection: newlines and tabs are collapsed and each
        # action is length-capped so a document-sized literal argument (e.g. a word ``write_to_file``
        # content) cannot bloat or corrupt the committed-action record.  The same deterministic
        # transform is applied on both sides wherever signatures are compared, so identical
        # trajectories still compare equal.
        parts = []
        for a in self.actions:
            s = str(a).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            parts.append(s[:400] + "…" if len(s) > 400 else s)
        return " | ".join(parts)

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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    n_python_calls: int = 0
    n_python_failures: int = 0
    n_python_forced: int = 0
    n_python_mode_misses: int = 0
    score: float = 0.0


__all__ = ["Observation", "Step", "Trajectory", "RewardInfo", "RolloutResult"]
