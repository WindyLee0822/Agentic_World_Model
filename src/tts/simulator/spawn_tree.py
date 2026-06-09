"""Spawn-tree logging: parent->child, region, depth, confidence, per-node cost/tokens/reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpawnNode:
    """One node of the recursion (a simulator root or a probe subagent)."""

    node_id: str
    parent_id: Optional[str]
    depth: int
    region: str
    role: str  # "simulator" | "probe"
    confidence: float = 0.0
    n_reads: int = 0
    keys_covered: int = 0
    tokens: int = 0
    cost: float = 0.0
    spawned: bool = False
    success: Optional[bool] = None
    note: str = ""
    children: List["SpawnNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def max_depth(self) -> int:
        return max((n.depth for n in self.walk()), default=self.depth)

    def n_nodes(self) -> int:
        return sum(1 for _ in self.walk())

    def n_spawns(self) -> int:
        return sum(len(n.children) for n in self.walk())

    def depth_histogram(self) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for n in self.walk():
            hist[n.depth] = hist.get(n.depth, 0) + 1
        return hist

    def total_tokens(self) -> int:
        return sum(n.tokens for n in self.walk())

    def total_cost(self) -> float:
        return sum(n.cost for n in self.walk())

    def total_reads(self) -> int:
        return sum(n.n_reads for n in self.walk())

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "region": self.region,
            "role": self.role,
            "confidence": round(self.confidence, 4),
            "n_reads": self.n_reads,
            "keys_covered": self.keys_covered,
            "tokens": self.tokens,
            "cost": self.cost,
            "spawned": self.spawned,
            "success": self.success,
            "note": self.note,
            "children": [c.to_dict() for c in self.children],
        }

    def render_ascii(self, indent: int = 0) -> str:
        pad = "  " * indent
        line = (
            f"{pad}- [{self.role} d{self.depth}] {self.region} conf={self.confidence:.2f} "
            f"reads={self.n_reads} cov={self.keys_covered} tok={self.tokens}"
            + (" *spawn" if self.spawned else "")
        )
        out = [line]
        for c in self.children:
            out.append(c.render_ascii(indent + 1))
        return "\n".join(out)


__all__ = ["SpawnNode"]
