#!/usr/bin/env python3
"""Summarize a dw ``records.jsonl`` into downloadable metric files (4 decimals).

dw's harness only PRINTS its metrics report to stdout (which on AML lands in the
system log, not the downloadable job output). This standalone helper re-reads the
checkpointed ``records.jsonl`` and writes:

  * ``summary.json`` — aggregates by (env, subset, method, n_rollouts) + the
    success-vs-N TTS curve, every float rounded to 4 decimals.
  * ``summary.txt``  — the same TTS curve as a human-readable table (4 decimals).

Pure stdlib so it runs under any of the project's conda envs. Mirrors the logic of
experiments/run.py ``aggregate``/``tts_table`` without importing the package.

Usage: python summarize.py <records.jsonl> <out_dir> [--strict-coverage]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ND = 4  # decimal places for every reported metric
_METHOD_ORDER = {"b0": 0, "b1": 1, "b2": 2, "ours": 3}


def _round(x):
    return round(x, ND) if isinstance(x, (int, float)) else x


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _load(path: Path):
    recs = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def aggregate(records, by):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r.get(k) for k in by)].append(r)
    rows = []
    for key, rs in groups.items():
        row = dict(zip(by, key))
        preds = [r["prediction_correct"] for r in rs if r.get("prediction_correct") is not None]
        imputed_failures = sum(bool(r.get("coverage_imputed_failure")) for r in rs)
        row.update({
            "decisions": len(rs),
            "recorded_decisions": len(rs) - imputed_failures,
            "imputed_failures": imputed_failures,
            "success_rate": _round(_mean([bool(r.get("success")) for r in rs])),
            "reward": _round(_mean([r.get("reward") for r in rs])),
            "prediction_accuracy": _round(statistics.fmean(preds)) if preds else None,
            "single_commit_rate": _round(_mean([r.get("n_commits") == 1 for r in rs])),
            "mean_spawns": _round(_mean([r.get("n_spawns") for r in rs])),
            "mean_reads": _round(_mean([r.get("n_reads") for r in rs])),
            "mean_tokens": _round(_mean([r.get("tokens") for r in rs])),
            "mean_cost": _round(_mean([r.get("cost") for r in rs])),
            "mean_wall_clock_s": _round(_mean([r.get("wall_clock_s") for r in rs])),
            "mean_rollout_errors": _round(_mean([r.get("n_rollout_errors", 0) for r in rs])),
            "simulator_writes": sum(r.get("simulator_writes") or 0 for r in rs),
        })
        rows.append(row)
    return rows


def _ordered_methods(records):
    return sorted({r["method"] for r in records}, key=lambda m: _METHOD_ORDER.get(m, 99))


def tts_table(records, title):
    Ns = sorted({r["n_rollouts"] for r in records})
    lut = {(r["method"], r["n_rollouts"]): r["success_rate"]
           for r in aggregate(records, ["method", "n_rollouts"])}
    out = [f"\n{title}", "-" * len(title), "  method | " + " ".join(f"N={n:<8}" for n in Ns)]
    for m in _ordered_methods(records):
        if m == "b0":
            base = next((lut[k] for k in lut if k[0] == "b0"), float("nan"))
            vals = [base] * len(Ns)
        else:
            vals = [lut.get((m, n), float("nan")) for n in Ns]
        out.append(f"  {m:>6} | " + " ".join(f"{(v if v is not None else float('nan')):<10.4f}" for v in vals))
    return "\n".join(out)


def _expected_cells(manifest, records):
    cells = {
        (str(cell["method"]), int(cell["n_rollouts"]))
        for cell in manifest.get("expected_cells", [])
        if isinstance(cell, dict) and cell.get("method") is not None
        and cell.get("n_rollouts") is not None
    }
    if not cells:
        cells = {
            (str(record["method"]), int(record["n_rollouts"]))
            for record in records
            if record.get("method") is not None and record.get("n_rollouts") is not None
        }
    expected_count = int(manifest.get("expected_cells_per_task", len(cells)) or 0)
    if not cells or len(cells) != expected_count:
        raise ValueError(
            f"cannot identify all expected cells: found {len(cells)}, expected {expected_count}"
        )
    return cells


def _records_with_missing_failures(records, manifest):
    expected_cells = _expected_cells(manifest, records)
    by_subset = manifest.get("by_subset")
    if not isinstance(by_subset, dict) or not by_subset:
        raise ValueError("run manifest has no per-subset coverage")

    actual_counts = defaultdict(int)
    metric_env_by_subset = {}
    for record in records:
        subset = record.get("spec_subset", record.get("subset"))
        if subset is None:
            continue
        actual_counts[(subset, str(record.get("method")), record.get("n_rollouts"))] += 1
        if record.get("env"):
            metric_env_by_subset.setdefault(subset, record["env"])

    metric_records = list(records)
    manifest_env = str(manifest.get("env", ""))
    for subset, coverage in by_subset.items():
        expected_tasks = int(coverage.get("expected_tasks", 0) or 0)
        if expected_tasks <= 0:
            raise ValueError(f"subset {subset!r} has no expected tasks")
        metric_env = metric_env_by_subset.get(subset)
        if metric_env is None:
            metric_env = f"tau2_{subset}" if manifest_env == "tau2" else manifest_env
        for method, n_rollouts in expected_cells:
            actual = actual_counts[(subset, method, n_rollouts)]
            if actual > expected_tasks:
                raise ValueError(
                    f"subset {subset!r} cell {method}/N={n_rollouts} has "
                    f"{actual} records but only {expected_tasks} expected tasks"
                )
            for _ in range(expected_tasks - actual):
                metric_records.append({
                    "env": metric_env,
                    "subset": subset,
                    "spec_subset": subset,
                    "method": method,
                    "n_rollouts": n_rollouts,
                    "success": False,
                    "reward": 0.0,
                    "n_commits": 0,
                    "n_rollout_errors": None,
                    "coverage_imputed_failure": True,
                })
    return metric_records


def main() -> int:
    args = sys.argv[1:]
    coverage_flags = {"--allow-partial", "--strict-coverage"}.intersection(args)
    if len(coverage_flags) > 1:
        print("--allow-partial and --strict-coverage are mutually exclusive", file=sys.stderr)
        return 2
    strict_coverage = "--strict-coverage" in coverage_flags
    args = [arg for arg in args if arg not in coverage_flags]
    if len(args) != 2:
        print(
            "usage: python summarize.py <records.jsonl> <out_dir> "
            "[--allow-partial | --strict-coverage]",
            file=sys.stderr,
        )
        return 2
    records_path, out_dir = map(Path, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records_path.is_file():
        message = f"[summarize] records are missing at {records_path}; coverage is unknown"
        (out_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"error": "missing_records", "records": str(records_path)}, indent=2),
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        return 2
    records = _load(records_path)
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.is_file():
        message = f"[summarize] {manifest_path} is missing; refusing to summarize unknown coverage"
        (out_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"error": "missing_run_manifest"}, indent=2),
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        message = f"[summarize] invalid run manifest at {manifest_path}: {error}"
        (out_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"error": "invalid_run_manifest"}, indent=2),
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        return 2
    expected_tasks = int(manifest.get("expected_tasks", 0) or 0)
    incomplete = int(manifest.get("partial_tasks", 0)) + int(manifest.get("missing_tasks", 0))
    if expected_tasks <= 0:
        message = (
            f"[summarize] invalid run manifest at {manifest_path}: "
            "expected_tasks must be positive"
        )
        (out_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"error": "invalid_run_manifest", "coverage": manifest}, indent=2),
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        return 2
    if incomplete and strict_coverage:
        message = (
            f"[summarize] refusing to publish an unequal-denominator curve: "
            f"{incomplete}/{expected_tasks} task(s) have missing cells. "
            f"See {manifest_path} and re-run with --resume, or omit --strict-coverage."
        )
        (out_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"error": "incomplete_coverage", "coverage": manifest}, indent=2),
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        return 2
    try:
        metric_records = _records_with_missing_failures(records, manifest)
    except (TypeError, ValueError) as error:
        message = f"[summarize] invalid coverage metadata: {error}"
        (out_dir / "summary.txt").write_text(message + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"error": "invalid_run_manifest", "coverage": manifest}, indent=2),
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        return 2

    imputed_failures = len(metric_records) - len(records)
    envs = sorted({r["env"] for r in metric_records})
    partial_warning = ""
    if incomplete:
        partial_warning = (
            f"PARTIAL RESULTS: {incomplete}/{expected_tasks} task(s) have missing cells. "
            f"All {imputed_failures} missing decision cell(s) count as failures. "
            "See recorded_decisions/imputed_failures and the embedded coverage manifest."
        )
        print(f"[summarize] {partial_warning}", file=sys.stderr)
    summary = {
        "decisions": len(metric_records),
        "recorded_decisions": len(records),
        "imputed_failures": imputed_failures,
        "partial": bool(incomplete),
        "coverage": manifest,
        "by_method_n": aggregate(metric_records, ["env", "subset", "method", "n_rollouts"]),
        "by_method_n_overall": aggregate(metric_records, ["env", "method", "n_rollouts"]),
    }
    if partial_warning:
        summary["warning"] = partial_warning
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [partial_warning, ""] if partial_warning else []
    for env_name in envs:
        env_recs = [r for r in metric_records if r["env"] == env_name]
        lines.append(tts_table(env_recs, f"TTS curve — success vs N  [{env_name} | ALL subsets]"))
        for sub in sorted({r["subset"] for r in env_recs}):
            lines.append(tts_table([r for r in env_recs if r["subset"] == sub],
                                   f"TTS — success vs N  [{env_name} | {sub}]"))
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")

    # Also PRINT the 4-decimal tables to stdout so they show on the command line
    # (the harness itself only prints 2 decimals). The wrapper tees this to run.log.
    print("\n" + "=" * 60)
    print("METRICS (4 decimal places)")
    print("=" * 60)
    print(text, end="")
    print(f"[summarize] wrote {out_dir/'summary.json'} and {out_dir/'summary.txt'} (4-decimal metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
