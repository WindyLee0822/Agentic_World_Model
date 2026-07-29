"""Offline end-to-end test of the optimization loop (no network).

A scripted LLM makes the baseline verifier imperfect and a specific reworded template perfect, so we
can assert the optimizer (a) measures the baseline, (b) proposes+accepts the better template, (c)
always retains the baseline for fallback, and (d) round-trips through the store.

Run:  python -m pytest promptopt/tests/test_offline_loop.py -q
  or: python promptopt/tests/test_offline_loop.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))          # repo root (src on path via install)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))   # tts source

from promptopt.dataset import LabeledExample, SubsetSpec, slice_counts       # noqa: E402
from promptopt.evaluate import evaluate_template                             # noqa: E402
from promptopt import templates as T                                          # noqa: E402
from promptopt import optimize as O                                           # noqa: E402


class ScriptedLLM:
    """Verifier: perfect only when the prompt contains the token 'ORACLE' (a stand-in for a decisive
    decision rule the optimizer must add); otherwise it always guesses FAILURE.  Optimizer: injects
    'ORACLE' into the system + every text section, turning an imperfect prompt into a perfect one."""

    MARK_OK = "[y]"   # embedded in example data for a true success
    MARK_NO = "[n]"

    def chat(self, system, user, *, temperature=0.0, max_tokens=None):
        text = f"{system}\n{user}"
        if "text_sections" in text:  # optimizer call
            n = text.count("[text_section ")
            return json.dumps({
                "lessons": "add a decisive rule",
                "system": (system_hint := "ORACLE strict judge"),
                "text_sections": ["ORACLE " + f"rule {i}" for i in range(n)],
            })
        # verifier call
        oracle = "ORACLE" in text
        truth = self.MARK_OK in text
        if oracle:
            good = truth
        else:
            good = False  # baseline always says FAILURE/FAIL
        if "VERDICT" in text:
            return "VERDICT: SUCCESS" if good else "VERDICT: FAIL"
        return "SUCCESS" if good else "FAILURE"


def _make_examples():
    exs = []
    # OfficeBench: all three app-levels, mixed labels.
    for app in ("1app", "2app", "3app"):
        for j, truth in enumerate([True, True, False, False, True]):
            mark = ScriptedLLM.MARK_OK if truth else ScriptedLLM.MARK_NO
            exs.append(LabeledExample(
                env="officebench", subset=app, task_id=f"{app}-{j}", seed=j,
                data={"instruction": f"task {app}-{j} {mark}", "trajectory": "  step 0: x -> [ok]",
                      "final_state": f"  === /f ===\n  {mark}", "giveup_note": "",
                      "requirements": "  - check", "gold_files": ""},
                true_label=truth))
    # τ²: two domains, mixed labels.
    for dom in ("airline", "telecom"):
        for j, truth in enumerate([True, False, True, False]):
            mark = ScriptedLLM.MARK_OK if truth else ScriptedLLM.MARK_NO
            exs.append(LabeledExample(
                env="tau2", subset=dom, task_id=f"{dom}-{j}", seed=j,
                data={"domain": dom, "ref": "", "facts": f"rec {mark}",
                      "gold": "book(...)", "cand": f"book(...) {mark}"},
                true_label=truth))
    return exs


def test_subset_spec():
    exs = _make_examples()
    spec = SubsetSpec()
    assert spec.satisfied(exs), spec.missing(exs)
    # Drop 3app -> requirement violated.
    partial = [e for e in exs if e.slice_key != "officebench/3app"]
    assert not spec.satisfied(partial)
    assert "3app" in spec.missing(partial).get("officebench", [])


def test_baseline_imperfect_then_optimized():
    exs = _make_examples()
    llm = ScriptedLLM()
    for env in ("officebench", "tau2"):
        env_exs = [e for e in exs if e.env == env]
        base = T.baseline(env)
        base_m = evaluate_template(base, env_exs, llm)
        assert base_m.balanced_accuracy < 1.0, f"{env} baseline should be imperfect"

        res = O.optimize(env, env_exs, llm, llm, O.OptimizerConfig(rounds=2, max_workers=1),
                         log=lambda *_: None)
        assert res.best_metrics.balanced_accuracy >= base_m.balanced_accuracy
        assert res.best_metrics.balanced_accuracy == 1.0, f"{env} optimized should be perfect"
        assert res.improved
        # Baseline object is always retained unchanged for fallback.
        assert res.baseline.version == 0 and res.baseline.text_values() == base.text_values()

        with tempfile.TemporaryDirectory() as d:
            O.save_result(res, d)
            loaded = O.load_best(env, d)
            assert loaded is not None and loaded.text_values() == res.best.text_values()
            assert os.path.exists(os.path.join(d, f"{env}_baseline.json"))


if __name__ == "__main__":
    test_subset_spec()
    test_baseline_imperfect_then_optimized()
    print("slice_counts:", slice_counts(_make_examples()))
    print("OK: offline optimization loop passed for officebench + tau2")
