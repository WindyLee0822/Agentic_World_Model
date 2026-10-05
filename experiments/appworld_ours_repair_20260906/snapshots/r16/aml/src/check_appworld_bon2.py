#!/usr/bin/env python3
"""Validate the GPT-4.1 AppWorld Python-only multi-app BoN@2 score gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "test_normal/L2": 0.70,
    "test_challenge/L2": 0.60,
    "test_normal/L3": 0.63,
    "test_challenge/L3": 0.60,
}

EXPECTED_DECISIONS = {
    "test_normal/L2": 48,
    "test_challenge/L2": 150,
    "test_normal/L3": 63,
    "test_challenge/L3": 195,
}


def evaluate(records):
    relevant = [
        record for record in records
        if record.get("env") == "appworld"
        and record.get("method") == "ours"
        and record.get("n_rollouts") == 2
    ]
    modes = sorted({
        record.get("appworld_wm_tool_mode", "missing")
        for record in relevant
    })
    agents = sorted({
        record.get("solver_agent", "missing")
        for record in relevant
    })
    models = sorted({
        record.get("llm_agent", "missing")
        for record in relevant
    })
    precision_prompts = sorted({
        bool(record.get("appworld_precision_prompt", False))
        for record in relevant
    })
    agent_references = sorted({
        bool(record.get("appworld_agent_reference", False))
        for record in relevant
    })
    deterministic_dirty_reads = sorted({
        bool(record.get("appworld_deterministic_dirty_reads", False))
        for record in relevant
    })
    percheck_verifiers = sorted({
        bool(record.get("appworld_verifier_percheck", False))
        for record in relevant
    })
    success_scores = sorted({
        record.get("success_score", "missing")
        for record in relevant
    })
    grouped = {subset: [] for subset in THRESHOLDS}
    for record in relevant:
        subset = record.get("subset")
        if subset in grouped:
            grouped[subset].append(bool(record.get("success")))

    rows = []
    passed = (
        modes == ["python"]
        and agents == ["llm_wm"]
        and models == ["gpt-4.1"]
        and precision_prompts == [True]
        and agent_references == [True]
        and deterministic_dirty_reads == [True]
        and percheck_verifiers == [True]
        and success_scores == ["simulator_confidence"]
    )
    for subset, threshold in THRESHOLDS.items():
        values = grouped[subset]
        rate = sum(values) / len(values) if values else None
        expected = EXPECTED_DECISIONS[subset]
        row_passed = (
            len(values) == expected
            and rate is not None
            and rate > threshold
        )
        passed = passed and row_passed
        rows.append({
            "subset": subset,
            "decisions": len(values),
            "expected_decisions": expected,
            "success_rate": rate,
            "threshold_exclusive": threshold,
            "passed": row_passed,
        })
    return {
        "gate": "appworld_python_multi_app_bon2",
        "method": "ours",
        "n_rollouts": 2,
        "tool_modes": modes,
        "solver_agents": agents,
        "models": models,
        "precision_prompts": precision_prompts,
        "agent_references": agent_references,
        "deterministic_dirty_reads": deterministic_dirty_reads,
        "percheck_verifiers": percheck_verifiers,
        "success_scores": success_scores,
        "passed": passed,
        "rows": rows,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python check_appworld_bon2.py <records.jsonl> <out_dir>", file=sys.stderr)
        return 2
    records_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if not records_path.is_file():
        print(f"[bon2-gate] records are missing at {records_path}", file=sys.stderr)
        return 2

    records = []
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    result = evaluate(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bon2_gate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "AppWorld GPT-4.1 Python-only multi-app BoN@2 gate",
        f"tool modes: {', '.join(result['tool_modes']) or '(none)'}",
        f"solver agents: {', '.join(result['solver_agents']) or '(none)'}",
        f"models: {', '.join(result['models']) or '(none)'}",
        f"precision prompt: {result['precision_prompts']}",
        f"agent reference contract: {result['agent_references']}",
        f"deterministic dirty reads: {result['deterministic_dirty_reads']}",
        f"per-check verifier: {result['percheck_verifiers']}",
        f"success scores: {', '.join(result['success_scores']) or '(none)'}",
    ]
    for row in result["rows"]:
        rate = (
            f"{100.0 * row['success_rate']:.2f}%"
            if row["success_rate"] is not None else "missing"
        )
        lines.append(
            f"{row['subset']}: {rate} > {100.0 * row['threshold_exclusive']:.2f}% "
            f"({'PASS' if row['passed'] else 'FAIL'}, "
            f"n={row['decisions']}/{row['expected_decisions']})"
        )
    lines.append("OVERALL: " + ("PASS" if result["passed"] else "FAIL"))
    text = "\n".join(lines) + "\n"
    (out_dir / "bon2_gate.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
