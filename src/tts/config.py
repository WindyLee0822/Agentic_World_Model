"""Typed configuration with every ablation knob, plus YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SolverConfig:
    #: solver kind: "llm" (real official-prompt agent on the REAL env, reproduces the benchmark) |
    #: "llm_wm" (the SAME official agent + user simulator, but tool execution routed through the
    #: read-grounded world model: reads exact, writes predicted, real env untouched; the selected
    #: trajectory is then committed to the real env) | "scripted" (offline, LLM-free plan-following
    #: stub).  The experiment runner (experiments/run.py) defaults to "llm"; the library default is
    #: the offline stub so the env-agnostic unit tests need no benchmark packages or API access.
    agent: str = "scripted"
    #: model the official benchmark is scored with (Azure deployment / litellm suffix)
    llm_agent: Optional[str] = None
    #: user-simulator model for τ² (defaults to ``llm_agent``)
    llm_user: Optional[str] = None
    #: sampling temperature for multi-rollout diversity (B1/B2/Ours)
    temperature: float = 0.7
    #: max agent steps per rollout (cohering to the official runners)
    max_steps: int = 100
    #: retries after a WM-interactive rollout raises before producing a candidate
    rollout_retries: int = 2
    #: probability a *scripted* rollout follows the plan (offline stub only)
    p_correct: float = 0.0


@dataclass
class SimulatorConfig:
    #: "flat" (B1) | "snapshot" (B2) | "recursive" (Ours) | "none" (B0)
    kind: str = "recursive"
    #: predict every agent-visible action, including clean reads. Flat B1 uses its fixed context;
    #: recursive Ours may use its existing bounded read-only grounding tools.
    predict_all_actions: bool = True
    #: "uncertainty" (spawn while conf < tau) | "fixed" (spawn to depth L)
    recursion_mode: str = "uncertainty"
    tau: float = 0.9
    L: int = 2
    hard_cap: int = 6
    branching: int = 2
    #: max keys a single node may read ("no full state in context")
    node_read_limit: int = 1
    caching: bool = True
    selective_invalidation: bool = True


@dataclass
class BudgetConfig:
    max_spawns: Optional[int] = 1024
    max_cost: Optional[float] = None
    max_reads: Optional[int] = None
    tokens_per_node: int = 100
    cost_per_token: float = 1e-6


@dataclass
class Config:
    method: str = "ours"  # b0 | b1 | b2 | ours
    env: str = "tau2"  # appworld | tau2
    N: int = 8
    success_score: str = "simulator_success"
    seed: int = 0

    solver: SolverConfig = field(default_factory=SolverConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def from_dict(data: Dict[str, Any]) -> Config:
    data = dict(data or {})
    solver = SolverConfig(**(data.pop("solver", {}) or {}))
    simulator = SimulatorConfig(**(data.pop("simulator", {}) or {}))
    budget = BudgetConfig(**(data.pop("budget", {}) or {}))
    return Config(solver=solver, simulator=simulator, budget=budget, **data)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    import yaml  # local import: only needed when loading YAML configs

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    merged = _merge(Config().to_dict(), data)
    if overrides:
        merged = _merge(merged, overrides)
    return from_dict(merged)


__all__ = ["Config", "SolverConfig", "SimulatorConfig", "BudgetConfig", "from_dict", "load_config"]
