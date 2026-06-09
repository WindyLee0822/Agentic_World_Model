"""Core typed data structures for the read-grounded TTS system over real environments.

The unit of state the simulator reasons about is a list of **row keys** that a candidate
action depends on (e.g., the flight numbers of a reservation, or the user it belongs to).
A :class:`Region` is an index slice over that per-action key list, so recursion partitions the
action's relevant state without ever holding all of it in one context.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: A backing-store row key (opaque string, e.g. ``"songs:42"`` or ``"seg:0"``).
RowKey = str
Row = Dict[str, Any]


@dataclass(frozen=True)
class Action:
    """A candidate action proposed by a solver and predicted by a simulator."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.name, json.dumps(self.args, sort_keys=True, default=str))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}({json.dumps(self.args, sort_keys=True, default=str)})"


@dataclass(frozen=True)
class Region:
    """An index slice ``[lo, hi)`` over a candidate action's relevant-key list.

    Regions are the unit of recursion: :meth:`split` partitions ``[lo, hi)`` into ``branching``
    contiguous sub-regions (zero-width sub-regions are allowed, keeping ``fixed``-depth
    recursion deterministic regardless of how many keys fall in the range).
    """

    lo: int = 0
    hi: int = 0

    @property
    def width(self) -> int:
        return max(0, self.hi - self.lo)

    def clamp(self, n: int) -> "Region":
        return Region(max(0, self.lo), min(self.hi, n))

    def split(self, branching: int) -> List["Region"]:
        branching = max(1, branching)
        w = self.width
        out: List[Region] = []
        for i in range(branching):
            out.append(Region(self.lo + (w * i) // branching, self.lo + (w * (i + 1)) // branching))
        return out

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.lo}:{self.hi}]"


@dataclass
class Summary:
    """A compact fold of read values that a probe returns to its parent.

    Carries enough to evaluate ``sum_le`` (``total``) and ``all_ge`` (``minimum``) predicates
    without holding the raw rows — this is what flows *up* the recursion.
    """

    covered: set = field(default_factory=set)
    total: float = 0.0
    minimum: float = math.inf
    count: int = 0

    def merged_with(self, other: "Summary") -> "Summary":
        return Summary(
            covered=self.covered | other.covered,
            total=self.total + other.total,
            minimum=min(self.minimum, other.minimum),
            count=self.count + other.count,
        )


@dataclass(frozen=True)
class Change:
    """A predicted or actual change to the environment."""

    target: str
    field: str
    old: Any
    new: Any


@dataclass
class Delta:
    """A predicted or actual effect of an action."""

    action: Action
    changes: List[Change] = field(default_factory=list)
    feasible: bool = True
    reason: str = ""

    def signature(self) -> str:
        body = sorted((c.target, c.field, repr(c.old), repr(c.new)) for c in self.changes)
        return json.dumps({"feasible": self.feasible, "changes": body}, sort_keys=True)


@dataclass
class Prediction:
    """What a simulator returns for one candidate action."""

    action: Action
    delta: Delta
    success: bool
    confidence: float
    spawn_root: Optional["object"] = None  # SpawnNode (avoid import cycle)
    n_reads: int = 0
    n_spawns: int = 0
    depth: int = 0
    tokens: int = 0
    cost: float = 0.0


@dataclass
class CommitResult:
    """Ground-truth outcome of committing exactly one action to the real environment."""

    action: Action
    delta: Delta
    success: bool
    reward: float


@dataclass
class Rollout:
    """One solver proposal plus its simulator evaluation."""

    index: int
    action: Action
    prediction: Optional[Prediction] = None
    score: float = 0.0


@dataclass
class DecisionRecord:
    """Everything logged about one decision (one commit) for TTS analysis."""

    method: str
    env: str
    subset: str
    task_id: str
    n_rollouts: int
    recursion_mode: Optional[str]
    selected_index: int
    committed_action: Optional[str]
    success: bool
    reward: float
    n_spawns: int = 0
    max_depth: int = 0
    depth_histogram: Dict[int, int] = field(default_factory=dict)
    n_reads: int = 0
    n_commits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens: int = 0
    cost: float = 0.0
    wall_clock_s: float = 0.0
    prediction_correct: Optional[bool] = None
    predicted_success: Optional[bool] = None
    simulator_writes: int = 0  # must always be 0 (simulator-never-writes invariant)
    split: Optional[str] = None
    level: Optional[int] = None
    domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["depth_histogram"] = {str(k): v for k, v in self.depth_histogram.items()}
        return d


__all__ = [
    "RowKey",
    "Row",
    "Action",
    "Region",
    "Summary",
    "Change",
    "Delta",
    "Prediction",
    "CommitResult",
    "Rollout",
    "DecisionRecord",
]
