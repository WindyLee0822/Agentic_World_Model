"""Global spawn/cost budget with a hard kill-switch (engineering guardrail)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


class BudgetExceeded(RuntimeError):
    """Raised when a spawn/cost limit is hit or the kill-switch is engaged."""


@dataclass
class Budget:
    """Shared counter for spawns / reads / tokens / cost across a whole decision."""

    max_spawns: int | None = None
    max_cost: float | None = None
    max_reads: int | None = None

    n_spawns: int = 0
    n_reads: int = 0
    tokens: int = 0
    cost: float = 0.0
    _killed: bool = field(default=False, repr=False)
    kill_switch_trips: int = 0

    def add_spawn(self, n: int = 1) -> None:
        self.n_spawns += n

    def add_read(self, n: int = 1) -> None:
        self.n_reads += n

    def add_cost(self, tokens: int, cost: float) -> None:
        self.tokens += tokens
        self.cost += cost

    def kill(self) -> None:
        self._killed = True

    @property
    def killed(self) -> bool:
        return self._killed

    def would_exceed_spawns(self, extra: int = 1) -> bool:
        return self.max_spawns is not None and (self.n_spawns + extra) > self.max_spawns

    def over_limit(self) -> bool:
        if self.max_spawns is not None and self.n_spawns > self.max_spawns:
            return True
        if self.max_cost is not None and self.cost > self.max_cost:
            return True
        if self.max_reads is not None and self.n_reads > self.max_reads:
            return True
        return False

    def should_stop(self, extra_spawn: int = 0) -> bool:
        """True if the kill-switch is engaged or a limit is (or would be) breached."""
        return self._killed or self.over_limit() or self.would_exceed_spawns(extra_spawn)

    def trip(self) -> None:
        self.kill_switch_trips += 1

    def check(self) -> None:
        if self.should_stop():
            self.trip()
            raise BudgetExceeded("budget exceeded or kill-switch engaged")

    def snapshot(self) -> Dict[str, float]:
        return {
            "n_spawns": self.n_spawns,
            "n_reads": self.n_reads,
            "tokens": self.tokens,
            "cost": self.cost,
            "kill_switch_trips": self.kill_switch_trips,
        }


__all__ = ["Budget", "BudgetExceeded"]
