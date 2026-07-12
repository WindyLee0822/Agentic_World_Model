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

Usage: python summarize.py <records.jsonl> <out_dir>
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
        row.update({
            "decisions": len(rs),
            "success_rate": _round(_mean([bool(r.get("success")) for r in rs])),
            "reward": _round(_mean([r.get("reward") for r in rs])),
            "prediction_accuracy": _round(statistics.fmean(preds)) if preds else None,
            "single_commit_rate": _round(_mean([r.get("n_commits") == 1 for r in rs])),
            "mean_spawns": _round(_mean([r.get("n_spawns") for r in rs])),
            "mean_reads": _round(_mean([r.get("n_reads") for r in rs])),
            "mean_tokens": _round(_mean([r.get("tokens") for r in rs])),
            "mean_cost": _round(_mean([r.get("cost") for r in rs])),
            "mean_wall_clock_s": _round(_mean([r.get("wall_clock_s") for r in rs])),
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


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python summarize.py <records.jsonl> <out_dir>", file=sys.stderr)
        return 2
    records_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if not records_path.is_file():
        print(f"[summarize] no records at {records_path}; nothing to summarize", file=sys.stderr)
        return 0
    records = _load(records_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        (out_dir / "summary.txt").write_text("(no decisions)\n", encoding="utf-8")
        (out_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        return 0

    envs = sorted({r["env"] for r in records})
    summary = {
        "decisions": len(records),
        "by_method_n": aggregate(records, ["env", "subset", "method", "n_rollouts"]),
        "by_method_n_overall": aggregate(records, ["env", "method", "n_rollouts"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = []
    for env_name in envs:
        env_recs = [r for r in records if r["env"] == env_name]
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
