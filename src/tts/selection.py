"""Pluggable ``success_score`` rankers + the trajectory selector.

After ``N`` trajectories are sampled and scored by their world model (or the B2 oracle), the
selector ranks them with a named ``success_score`` and returns the single best — whose *whole*
action sequence is then committed to the real environment.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from .trajectory import RolloutResult

ScoreFn = Callable[[RolloutResult, Sequence[RolloutResult]], float]
_REGISTRY: Dict[str, ScoreFn] = {}


def register(name: str) -> Callable[[ScoreFn], ScoreFn]:
    def deco(fn: ScoreFn) -> ScoreFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_score_fn(name: str) -> ScoreFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown success_score {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_scores() -> List[str]:
    return sorted(_REGISTRY)


@register("simulator_success")
def _simulator_success(r: RolloutResult, _all: Sequence[RolloutResult]) -> float:
    """Rank purely by the world model's predicted success flag for the trajectory.

    Deliberately not tie-broken by anything the flat model could inflate: a flat world model
    that mis-predicts every multi-entity trajectory as a failure cannot discriminate, so the
    selector falls back to a deterministic index tiebreak — exactly the behaviour the recursive
    world model is meant to beat.
    """
    return 1.0 if (r.predicted and r.predicted.success) else 0.0


@register("self_consistency")
def _self_consistency(r: RolloutResult, allr: Sequence[RolloutResult]) -> float:
    """Predicted success, tie-broken by agreement (how many rollouts share this trajectory)."""
    sig = r.trajectory.signature()
    same = sum(1 for o in allr if o.trajectory.signature() == sig)
    base = same / max(1, len(allr))
    feas = 1.0 if (r.predicted and r.predicted.success) else 0.0
    return feas + 1e-2 * base


@register("probe_agreement")
def _probe_agreement(r: RolloutResult, _all: Sequence[RolloutResult]) -> float:
    """Predicted success, tie-broken by the read-coverage of the rollout's spawn tree."""
    feas = 1.0 if (r.predicted and r.predicted.success) else 0.0
    leaves = []
    if r.spawn_root is not None:
        leaves = [n for n in r.spawn_root.walk() if not n.children]
    frac = (sum(1.0 for n in leaves if n.success) / len(leaves)) if leaves else 0.0
    return feas + 1e-3 * frac


class Selector:
    """Scores rollouts with a named ranker and returns the best index."""

    def __init__(self, score_name: str = "simulator_success") -> None:
        self.score_name = score_name
        self.score_fn = get_score_fn(score_name)

    def select(self, rollouts: Sequence[RolloutResult]) -> Tuple[int, List[RolloutResult]]:
        scored = list(rollouts)
        for r in scored:
            r.score = self.score_fn(r, scored)
        best = max(
            range(len(scored)),
            key=lambda i: (scored[i].score, -scored[i].cost, -i),
        )
        return best, scored


__all__ = ["Selector", "register", "get_score_fn", "available_scores", "ScoreFn"]
