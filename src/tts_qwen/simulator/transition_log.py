"""Generic append-only transition log for predictive world models.

The real environment remains immutable during sampling. Predictors record successful and rejected
state-changing actions here so a tool-using world-model subagent can inspect their ordered effects
through a domain-neutral ``TRACE <resource>`` operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Transition:
    """One predicted state transition and the resources it may affect."""

    action: Any
    observation: str
    resources: Tuple[str, ...]
    applied: bool


class TransitionLog:
    """Append-only record of predicted transitions, indexed by logical resource."""

    def __init__(self) -> None:
        self._entries: List[Transition] = []

    def record(self, action: Any, observation: str, resources: Iterable[str],
               *, applied: bool) -> Transition:
        normalized = tuple(dict.fromkeys(str(r) for r in resources if r))
        entry = Transition(action, str(observation), normalized, bool(applied))
        self._entries.append(entry)
        return entry

    def entries(self, resource: Optional[str] = None, *, applied_only: bool = False) -> List[Transition]:
        out = self._entries
        if resource is not None:
            out = [entry for entry in out if resource in entry.resources]
        if applied_only:
            out = [entry for entry in out if entry.applied]
        return list(out)

    def resources(self, *, applied_only: bool = True) -> List[str]:
        seen = {}
        for entry in self.entries(applied_only=applied_only):
            for resource in entry.resources:
                seen.setdefault(resource, None)
        return list(seen)

    def render(self, resource: Optional[str] = None, *, cap: int = 40) -> str:
        entries = self.entries(resource)[:cap]
        if not entries:
            return "  (no recorded transitions)"
        lines = []
        for entry in entries:
            status = "APPLIED" if entry.applied else "REJECTED"
            action = str(entry.action).replace("\n", "\\n")[:500]
            observation = entry.observation.replace("\n", "\\n")[:200]
            lines.append(f"  - {status} {action} -> [{observation}]")
        return "\n".join(lines)


__all__ = ["Transition", "TransitionLog"]
