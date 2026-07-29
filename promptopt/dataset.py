"""Labeled-example schema, JSONL store, and the eval-subset spec.

A :class:`LabeledExample` captures everything needed to (a) rebuild the verifier prompt with any
candidate template and (b) score it against the real-environment truth:

* ``data`` — the pre-rendered *data slots* the template interpolates (instruction, trajectory,
  predicted state, gold checks, …).  Only wording lives in the template; data lives here.
* ``true_label`` — the real-environment grounded success (the optimization target).
* ``baseline_pred`` — the current prompt's prediction at collection time (for reference).

The **eval subset** must span all OfficeBench app-levels (1/2/3-app) and ≥2 τ² domains, per the task
requirement; :class:`SubsetSpec` encodes and checks that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence


@dataclass
class LabeledExample:
    env: str                      # "officebench" | "tau2"
    subset: str                   # OB: "1app"/"2app"/"3app"; τ²: domain ("airline", …)
    task_id: str
    seed: int
    data: Dict[str, str]          # pre-rendered data slots consumed by the template
    true_label: bool              # real-env grounded success
    baseline_pred: float = 0.0    # current-prompt prediction at collection time
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def slice_key(self) -> str:
        return f"{self.env}/{self.subset}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "LabeledExample":
        d = json.loads(line)
        return LabeledExample(
            env=d["env"], subset=d["subset"], task_id=str(d["task_id"]), seed=int(d["seed"]),
            data=dict(d["data"]), true_label=bool(d["true_label"]),
            baseline_pred=float(d.get("baseline_pred", 0.0)), meta=dict(d.get("meta", {})),
        )


def save_examples(path: str, examples: Sequence[LabeledExample]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")


def append_examples(path: str, examples: Sequence[LabeledExample]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")


def load_examples(path: str, env: Optional[str] = None) -> List[LabeledExample]:
    out: List[LabeledExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = LabeledExample.from_json(line)
            if env is None or ex.env == env:
                out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Eval-subset composition requirement
# ---------------------------------------------------------------------------

#: OfficeBench app-levels the subset must cover.
OFFICEBENCH_REQUIRED = ("1app", "2app", "3app")
#: τ² domains the subset must cover (≥2); airline + telecom are both world-model-eligible.
TAU2_REQUIRED_MIN = 2
TAU2_DEFAULT_DOMAINS = ("airline", "telecom")


@dataclass
class SubsetSpec:
    """The composition an eval subset must satisfy."""

    officebench_apps: Sequence[str] = OFFICEBENCH_REQUIRED
    tau2_domains: Sequence[str] = TAU2_DEFAULT_DOMAINS

    def missing(self, examples: Sequence[LabeledExample]) -> Dict[str, List[str]]:
        """Return required slices with no examples, grouped by env (empty ⇒ satisfied)."""
        have = {ex.slice_key for ex in examples}
        miss: Dict[str, List[str]] = {"officebench": [], "tau2": []}
        for a in self.officebench_apps:
            if f"officebench/{a}" not in have:
                miss["officebench"].append(a)
        tau2_have = [d for d in self.tau2_domains if f"tau2/{d}" in have]
        if len(tau2_have) < min(TAU2_REQUIRED_MIN, len(self.tau2_domains)):
            miss["tau2"] = [d for d in self.tau2_domains if f"tau2/{d}" not in have]
        return {k: v for k, v in miss.items() if v}

    def satisfied(self, examples: Sequence[LabeledExample]) -> bool:
        return not self.missing(examples)


def slice_counts(examples: Sequence[LabeledExample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ex in examples:
        out[ex.slice_key] = out.get(ex.slice_key, 0) + 1
    return dict(sorted(out.items()))


def label_balance(examples: Sequence[LabeledExample]) -> Dict[str, int]:
    pos = sum(1 for e in examples if e.true_label)
    return {"total": len(examples), "success": pos, "fail": len(examples) - pos}


__all__ = [
    "LabeledExample", "save_examples", "append_examples", "load_examples",
    "SubsetSpec", "OFFICEBENCH_REQUIRED", "TAU2_DEFAULT_DOMAINS",
    "slice_counts", "label_balance",
]
