"""Check full-coverage, simulator-selected AppWorld Ours BoN@2 > 60%.

This audits recorded results; source and runtime routing must also be reviewed.
The reference records supply task identities only, never success labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


EXPECTED_COUNTS = {
    "test_normal/L1": 57, "test_normal/L2": 48, "test_normal/L3": 63,
    "test_challenge/L1": 72, "test_challenge/L2": 150, "test_challenge/L3": 195,
}


def _target_rows(records):
    return [r for r in records if r.get("env") == "appworld"
            and r.get("method") == "ours" and r.get("n_rollouts") == 2]


def evaluate(records, reference_records, *, require_setup_predictions=False):
    reference = _target_rows(reference_records)
    expected = {(r.get("subset"), r.get("task_id")) for r in reference}
    expected_counts = Counter(subset for subset, task_id in expected if task_id)
    if dict(expected_counts) != EXPECTED_COUNTS:
        raise ValueError("Reference identities must cover all 585 official test tasks")
    rows = _target_rows(records)
    keys = [(r.get("subset"), r.get("task_id")) for r in rows]
    counts = Counter(keys)
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected, key=str)
    duplicates = [key for key, count in counts.items() if count > 1]
    violations = []
    for row in rows:
        reasons = []
        required = {
            "solver_agent": "llm_wm", "predict_all_actions": True,
            "simulation_only": False, "agent_self_aggregate": False,
            "simulator_writes": 0, "appworld_agent_reference": False,
            "appworld_instruction_preflights": False,
            "appworld_preground_clean_reads": False,
            "appworld_deterministic_dirty_reads": False,
            "appworld_deterministic_grounded_clean_reads": False,
            "appworld_predict_all_program_only": True,
        }
        for field, value in required.items():
            if field not in row or row[field] != value:
                reasons.append(field)
        if row.get("success_score") not in {
            "simulator_success", "simulator_confidence", "simulator_confidence_guarded",
        }:
            reasons.append("success_score")
        if row.get("candidate_grounded_success") != [None, None]:
            reasons.append("candidate_grounded_success")
        if require_setup_predictions:
            setup = row.get("candidate_setup_prediction")
            if (not isinstance(setup, list) or len(setup) != 2
                    or any(not isinstance(item, dict) or item.get("enabled") is not True
                           or type(item.get("count")) is not int or item["count"] < 0
                           for item in setup)):
                reasons.append("candidate_setup_prediction")
        commits = row.get("n_commits")
        if commits not in (0, 1) or (row.get("success") is True and commits != 1):
            reasons.append("n_commits")
        if not isinstance(row.get("success"), bool):
            reasons.append("success")
        if reasons:
            violations.append({"task_id": row.get("task_id"), "fields": reasons})
    successes = sum(row.get("success") is True for row in rows)
    total = sum(EXPECTED_COUNTS.values())
    complete = not (missing or extra or duplicates)
    return {
        "goal": "overall AppWorld Ours BoN@2 > 60%",
        "passed": complete and not violations and successes * 5 > total * 3,
        "full_coverage": complete, "expected_decisions": total,
        "observed_decisions": len(rows), "official_successes": successes,
        "required_successes": 352,
        "success_rate": successes / total if complete else None,
        "missing": missing, "extra": extra, "duplicates": duplicates,
        "contract_violations": violations,
        "by_subset": {
            subset: {"decisions": sum(r.get("subset") == subset for r in rows),
                     "successes": sum(r.get("subset") == subset and r.get("success") is True
                                      for r in rows), "expected": count}
            for subset, count in EXPECTED_COUNTS.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--require-setup-predictions", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    reference = [json.loads(line) for line in args.reference_records.read_text().splitlines() if line.strip()]
    result = evaluate(records, reference, require_setup_predictions=args.require_setup_predictions)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
