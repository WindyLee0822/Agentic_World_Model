#!/usr/bin/env python3
"""Plot the OfficeBench token estimates without substituting measured-usage claims.

Requires matplotlib. Run from any directory:
    python draw_token_consumption.py
    python draw_token_consumption.py --width 10 --height 3.8 --output-dir figures

Reads summary.jsonl (a single JSON object); exports PDF (embedded fonts), SVG, 600-dpi PNG, the plotted
values as CSV, and a suggested LaTeX caption. No LLM calls or data modification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
INPUT_COLOR = "#4C708C"
OUTPUT_COLOR = "#D6B778"
INK = "#28323B"

COMPARISON = [
    ("ours", "Ours", 4.2, "GPT-5.4"),
    ("b1", "SIMIA", 3.2, "GPT-5.4"),
    ("ours_qwen", "Ours", 1.5, "Qwen"),
    ("b1_qwen", "SIMIA", 0.5, "Qwen"),
    ("agentworld", "WM-Training", -0.5, "Qwen"),
]

ABLATION = [
    ("ours", "Ours", 3.0, "GPT-5.4"),
    ("ours_no_context_management",
     "w.o. subagent", 2.0, "GPT-5.4"),
    ("ours_qwen", "Ours", 1.0, "Qwen"),
    ("ours_qwen_no_context_management",
     "w.o. subagent", 0.0, "Qwen"),
]


def read_data(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {m for m, _, _, _ in COMPARISON + ABLATION}
    for method in required:
        if data.get(method) is None:
            continue
        block = data[method]
        values = [block[s]["total"]["mean"] for s in ("low", "central", "high")]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f"Invalid token totals: {method}")
        if not values[0] <= values[1] <= values[2]:
            raise ValueError(f"Scenarios do not bracket central estimate: {method}")
        c = block["central"]
        if not math.isclose(c["input"]["mean"] + c["output"]["mean"], c["total"]["mean"]):
            raise ValueError(f"Component totals disagree: {method}")
        counts = [block[s]["total"]["n"] for s in ("low", "central", "high")]
        if not 0 < counts[0] <= 300 or len(set(counts)) != 1:
            raise ValueError(f"Invalid or inconsistent task coverage: {method}")
        if counts[0] != 300 and not block.get("weighting", block.get("metadata", {}).get("weighting")):
            raise ValueError(f"Partial coverage needs explicit weighting: {method}")
    return data


def panel(ax, data, entries, *, ratios=False):
    """Stacked columns, grouped by model family on a linear token axis."""
    positions = []
    groups = {}
    x = 0.0
    previous_family = None
    for method, label, _, family in entries:
        if previous_family is not None and family != previous_family:
            x += .4
        positions.append(x)
        groups.setdefault(family, []).append(x)
        previous_family = family
        x += 1

    high = max((data[m]["high"]["total"]["mean"] / 1000
                for m, _, _, _ in entries if data.get(m)), default=100)
    step = 50 if high < 250 else 400
    rounding = 50 if high < 250 else 100
    limit = math.ceil(high * 1.08 / rounding) * rounding
    for x, (method, label, _, family) in zip(positions, entries):
        if not data.get(method):
            ax.text(x, limit * .025, "Pending", ha="center", fontsize=10)
            continue
        block = data[method]
        c = block["central"]
        mean, inp, out = (c[k]["mean"] / 1000 for k in ("total", "input", "output"))
        lo, hi = (block[k]["total"]["mean"] / 1000 for k in ("low", "high"))
        ax.bar(x, inp, width=.68, color=INPUT_COLOR, zorder=3)
        ax.bar(x, out, bottom=inp, width=.68, color=OUTPUT_COLOR,
               edgecolor="white", linewidth=.35, zorder=3)
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                    ecolor=INK, elinewidth=.8, capsize=3, capthick=.8, zorder=4)
        ax.annotate(f"{mean:.1f}k", (x, hi), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11)
        if ratios and method not in ("ours", "ours_qwen"):
            base = data["ours" if family == "GPT-5.4" else "ours_qwen"]["central"]["total"]["mean"]
            ax.text(x, mean * .45, f"{c['total']['mean'] / base:.2f}×",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", zorder=5)

    ax.set_xticks(positions, [label for _, label, _, _ in entries])
    for tick, (method, _, _, _) in zip(ax.get_xticklabels(), entries):
        if method in ("ours", "ours_qwen"):
            tick.set_fontweight("bold")
    for family, xs in groups.items():
        ax.plot([xs[0] - .4, xs[-1] + .4], [-.14, -.14],
                transform=ax.get_xaxis_transform(), color="#A0A8AF",
                linewidth=.7, clip_on=False)
        ax.text(sum(xs) / len(xs), -.19, family, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12, fontweight="bold")
    ax.set_xlim(positions[0] - .6, positions[-1] + .6)
    ax.set_ylim(0, limit)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_ylabel("Mean tokens per task (×10³)", fontsize=11, labelpad=5)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=10)
    ax.tick_params(axis="y", length=2.5, color="#A0A8AF", labelsize=10)
    ax.grid(axis="y", color="#E8EBED", linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#A0A8AF")
        ax.spines[spine].set_linewidth(.6)


def export_table(data, output_dir):
    fields = ["method", "status", "n_tasks", "population_n_tasks", "weighting", "mean_input_tokens", "mean_output_tokens",
              "mean_total_tokens", "median_total_tokens", "p90_total_tokens",
              "low_scenario_mean_tokens", "high_scenario_mean_tokens"]
    with (output_dir / "officebench_token_usage_values.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in dict.fromkeys(
            m for m, _, _, _ in COMPARISON + ABLATION
        ):
            if data.get(method) is None:
                writer.writerow(dict(method=method, status="pending"))
                continue
            b = data[method]
            c = b["central"]
            writer.writerow(dict(method=method, status=b.get("status", b.get("metadata", {}).get("status", "estimated")), n_tasks=c["total"]["n"],
                population_n_tasks=b.get("population_n_tasks", b.get("metadata", {}).get("population_tasks", c["total"]["n"])), weighting=b.get("weighting", b.get("metadata", {}).get("weighting", "none")),
                mean_input_tokens=c["input"]["mean"], mean_output_tokens=c["output"]["mean"],
                mean_total_tokens=c["total"]["mean"], median_total_tokens=c["total"]["median"],
                p90_total_tokens=c["total"]["p90"], low_scenario_mean_tokens=b["low"]["total"]["mean"],
                high_scenario_mean_tokens=b["high"]["total"]["mean"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--width", type=float, default=9.2, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=3.3, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    data = read_data(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 12,
        "mathtext.fontset": "stix",
        "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height))
    fig.subplots_adjust(left=.075, right=.995, bottom=.21, top=.87, wspace=.25)
    panel(axes[0], data, COMPARISON)
    panel(axes[1], data, ABLATION, ratios=True)
    axes[0].set_title("(a) Baseline comparison", loc="left", fontsize=12,
                      fontweight="bold", pad=8)
    axes[1].set_title("(b) Subagent ablation", loc="left", fontsize=12,
                      fontweight="bold", pad=8)
    fig.legend(handles=[Patch(facecolor=INPUT_COLOR, label="Input tokens"),
                        Patch(facecolor=OUTPUT_COLOR, label="Output tokens")],
               loc="upper center", bbox_to_anchor=(.54, .045), ncol=2, frameon=False,
               fontsize=11, handlelength=1.5, handleheight=.75, columnspacing=2.4)
    ours_n = data["ours"]["central"]["total"]["n"] if data.get("ours") else 0
    qwen_b1_n = data["b1_qwen"]["central"]["total"]["n"] if data.get("b1_qwen") else 0
    # fig.text(.105, .045,
    #          f"N = 1. Weighted samples: GPT-5.4 Ours {ours_n}/300; Qwen B1 {qwen_b1_n}/300. Other runs: 300 tasks.\n"
    #          "Whiskers: low–high modeling scenarios. Ratios: relative to Ours within each model family.",
    #          fontsize=7.5, color="#596774", linespacing=1.5)
    base = args.output_dir / "officebench_token_usage"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(base.with_suffix("." + ext), dpi=args.dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    export_table(data, args.output_dir)
    #     caption = r"""\begin{figure*}[t]
    #   \centering
    #   \includegraphics[width=\textwidth]{officebench_token_usage.pdf}
    #   \caption{\textbf{Estimated token consumption on OfficeBench.}
    #   Estimated mean cumulative input and output tokens at N=1, including failures.
    #   GPT-5.4 Ours uses __GPT_N__ completed tasks, weighted to the benchmark's
    #   93/300, 95/300, and 112/300 one-, two-, and three-app proportions. This partial
    #   sample is not random and may underrepresent long unfinished trajectories.
    #   Qwen B1 uses __QWEN_B1_N__ sampled tasks, with population weighting across
    #   the same app-count strata; subset sampling uncertainty is separate from the whiskers.
    #   GPT-5.4 B1, Qwen Ours, and AgentWorld use all 300 tasks.
    #   Whiskers show low--high modeling scenarios, not confidence intervals.
    #   (a) GPT-5.4 Ours and B1 use saved AML trajectories with a 40-step limit.
    #   Qwen-family model variants and step limits differ.
    #   Ours uses Qwen-AgentWorld-35B-A3B; AgentWorld uses Qwen3.5-35B-A3B.
    #   (b) Ours without subagent management is the largest counterfactual context
    #   variant, retaining raw evidence, actor history, and prior world-model transcripts.
    #   Both bars use each model family's saved Ours trajectories, holding actions,
    #   modeled calls, and completions fixed; ratios are relative to Ours within
    #   the same model family.
    #   Counts use a common tokenizer proxy and modeled missing observations/outputs;
    #   these are not measured API usage or rerun ablations.}
    #   \label{fig:officebench-token-usage}
    # \end{figure*}
    # """
    #     caption = caption.replace("__GPT_N__", str(ours_n)).replace("__QWEN_B1_N__", str(qwen_b1_n))
    #     (args.output_dir / "officebench_token_usage_caption.tex").write_text(caption)
    print(f"Saved PDF, SVG, PNG, and plotted CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
