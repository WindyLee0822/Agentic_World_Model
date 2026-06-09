"""The single commit chokepoint: the one place a trajectory is written to the real env."""

from __future__ import annotations

from .abcs import TaskEnvironment
from .trajectory import RewardInfo, Trajectory


class CommitChokepoint:
    """Commits exactly one trajectory to the real environment, at most once.

    Every method routes its single real write through here, so the *single-commit* and
    *one-real-environment* invariants are enforced in one auditable place.
    """

    def __init__(self, env: TaskEnvironment) -> None:
        self._env = env
        self.used = False

    def commit(self, trajectory: Trajectory) -> RewardInfo:
        if self.used:
            raise RuntimeError("commit chokepoint already used: only one commit per decision")
        self.used = True
        return self._env.commit_trajectory(trajectory)


__all__ = ["CommitChokepoint"]
