"""Pluggable ``success_score`` rankers + the trajectory selector.

After ``N`` trajectories are sampled and scored by their world model (or the B2 oracle), the
selector ranks them with a named ``success_score`` and returns the single best — whose *whole*
action sequence is then committed to the real environment.
"""

from __future__ import annotations

import os
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


@register("simulator_confidence_guarded")
@register("simulator_confidence")
def _simulator_confidence(r: RolloutResult, _all: Sequence[RolloutResult]) -> float:
    """Rank by the world model's success CONFIDENCE (predicted reward) for the trajectory.

    The predicted ``reward`` may be a self-consistency vote fraction or a graded per-check audit, so
    this prefers the rollout the world model is most confident in — a sharper signal than binary
    ``simulator_success``. Reduces to ``simulator_success`` when reward is binary (0/1).
    """
    return float(r.predicted.reward) if r.predicted is not None else 0.0


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

        def is_hard_invalid(rollout: RolloutResult) -> bool:
            metadata = rollout.predicted.metadata if rollout.predicted is not None else {}
            return bool((metadata or {}).get("hard_invalid", False))

        def rank_key(i: int):
            # A rollout that exhausted generation retries remains in the pool so every N cell is
            # complete, but it must never beat a valid candidate merely because both score 0.
            # Likewise, a candidate that provably contradicts a native benchmark answer key cannot
            # win a zero-score tie against a candidate whose failure is merely uncertain.
            return (
                bool(scored[i].trajectory.done),
                not is_hard_invalid(scored[i]),
                scored[i].score,
                -scored[i].cost,
                -i,
            )

        if self.score_name != "simulator_confidence_guarded":
            best = max(range(len(scored)), key=rank_key)
            return best, scored

        overrides = [
            (rollout.predicted.metadata or {}).get("selection_override")
            for rollout in scored
            if rollout.predicted is not None
        ]
        current_overrides = [
            override for override in overrides
            if isinstance(override, dict)
            and int(override.get("pool_size", -1)) == len(scored)
        ]
        if current_overrides:
            best = int(current_overrides[0].get("best", -1))
            if not all(int(override.get("best", -1)) == best for override in current_overrides):
                raise ValueError("conflicting trajectory selection overrides")
            if not 0 <= best < len(scored):
                raise ValueError(
                    f"invalid trajectory selection override {best} for {len(scored)} rollouts"
                )
            return best, scored

        try:
            margin = max(
                0.0,
                float(os.environ.get(
                    "TTS_SELECTION_CONFIDENCE_MARGIN", "0.1")),
            )
        except (TypeError, ValueError):
            margin = 0.1

        # Compare nested BoN pools (1 -> 2 -> 4 -> ...). A later pool replaces the incumbent only
        # when it fixes validity or provides a clear confidence gain; small verifier noise cannot
        # turn a previously successful lower-N choice into a regression.
        best = 0
        start = 1
        end = min(2, len(scored))
        while start < len(scored):
            challenger = max(range(start, end), key=rank_key)
            incumbent_priority = rank_key(best)[:2]
            challenger_priority = rank_key(challenger)[:2]
            if (
                challenger_priority > incumbent_priority
                or (
                    challenger_priority == incumbent_priority
                    and scored[challenger].score > scored[best].score + margin
                )
            ):
                best = challenger
            start = end
            end = min(len(scored), max(start + 1, end * 2))
        return best, scored


__all__ = ["Selector", "register", "get_score_fn", "available_scores", "ScoreFn"]
