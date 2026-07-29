"""Score a prompt template against labeled examples.

The verifier's prediction is a confidence in ``[0, 1]`` (a K-vote success fraction); we threshold at
0.5 to a binary label and compare against the real-environment ``true_label``.  Because the two
classes (success / fail) are usually imbalanced, the default optimization objective is **balanced
accuracy** (mean of true-positive-rate and true-negative-rate); accuracy and F1 are also reported,
plus a per-slice breakdown so no OfficeBench app-level or τ² domain is silently ignored.
"""

from __future__ import annotations

import concurrent.futures as _fut
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .dataset import LabeledExample
from .templates import PromptTemplate, predict


@dataclass
class Metrics:
    n: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    per_slice: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)  # misclassified (for the optimizer)

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def tpr(self) -> float:
        p = self.tp + self.fn
        return self.tp / p if p else 1.0  # no positives ⇒ vacuously perfect on that class

    @property
    def tnr(self) -> float:
        n = self.tn + self.fp
        return self.tn / n if n else 1.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.tpr + self.tnr)

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom else 0.0

    def slice_accuracy(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, c in sorted(self.per_slice.items()):
            n = c["tp"] + c["tn"] + c["fp"] + c["fn"]
            out[k] = (c["tp"] + c["tn"]) / n if n else 0.0
        return out

    def summary(self) -> str:
        sl = ", ".join(f"{k}={v:.2f}" for k, v in self.slice_accuracy().items())
        return (f"n={self.n} acc={self.accuracy:.3f} bal_acc={self.balanced_accuracy:.3f} "
                f"f1={self.f1:.3f} (tp={self.tp} tn={self.tn} fp={self.fp} fn={self.fn}) "
                f"| slices: {sl}")

    def score(self, objective: str = "balanced_accuracy") -> float:
        return {
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "f1": self.f1,
        }.get(objective, self.balanced_accuracy)


def _blank_slice() -> Dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def evaluate_template(template: PromptTemplate, examples: Sequence[LabeledExample], llm, *,
                      k: int = 1, temperature: float = 0.7, threshold: float = 0.5,
                      max_workers: int = 1, keep_errors: bool = True,
                      max_tokens: int = 512) -> Metrics:
    """Evaluate ``template`` over ``examples`` using ``llm`` as the verifier."""
    m = Metrics()

    def _one(ex: LabeledExample):
        conf = predict(template, ex.data, llm, k=k, temperature=temperature, max_tokens=max_tokens)
        return ex, conf

    if max_workers > 1:
        with _fut.ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_one, examples))
    else:
        results = [_one(ex) for ex in examples]

    for ex, conf in results:
        pred = conf >= threshold
        truth = bool(ex.true_label)
        sl = m.per_slice.setdefault(ex.slice_key, _blank_slice())
        m.n += 1
        if pred and truth:
            m.tp += 1; sl["tp"] += 1
        elif not pred and not truth:
            m.tn += 1; sl["tn"] += 1
        elif pred and not truth:
            m.fp += 1; sl["fp"] += 1
            if keep_errors:
                m.errors.append({"kind": "false_positive", "ex": ex, "conf": conf})
        else:
            m.fn += 1; sl["fn"] += 1
            if keep_errors:
                m.errors.append({"kind": "false_negative", "ex": ex, "conf": conf})
    return m


__all__ = ["Metrics", "evaluate_template"]
