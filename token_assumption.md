# OfficeBench token-consumption approximation — 2026-09-07

> Current GPT-5.4 figure values use the collected-trajectory update at the end of this document; the September 7 tables below are historical.

Planning estimates from saved trajectories and reconstructed prompts, **not measured API usage**. One datum means one OfficeBench task/subtask, one candidate rollout (N=1), including actor, observation/final-state world model, and success verification. Tokens are cumulative input plus output over calls; repeated input counts each time. This is not peak context size or a dollar-cost estimate.

All 300 matched tasks are included: 93 one-app, 95 two-app, 112 three-app, including failures. No new LLM rollout was run. Pristine task snapshots were read in isolated Docker sessions; saved candidate writes were not executed.

## Estimates per task at N=1

| Approach | Mean | Median | SD | P90 | Low–high mean scenarios |
|---|---:|---:|---:|---:|---:|
| Ours (GPT-5.4 trajectories) | 34.6k | 21.6k | 38.6k | 70.9k | 27.1k–53.4k |
| B1 (GPT-5.4 trajectories) | 61.5k | 37.3k | 65.1k | 131.0k | 47.2k–94.4k |
| AgentWorld (Qwen3.5 trajectories) | 96.5k | 59.3k | 100.4k | 201.7k | 80.7k–138.1k |
| Ours, raw context + persistent WM transcripts | 172.4k | 50.8k | 376.2k | 386.7k | 87.6k–502.5k |
| Ours, raw context only | 47.9k | 27.2k | 62.6k | 98.5k | 33.2k–89.9k |
| Ours (Qwen-AgentWorld trajectories) | 72.8k | 35.6k | 77.3k | 203.6k | 61.0k–105.9k |

P90 describes the distribution of modeled per-task totals under the central scenario. Low/high are sensitivity scenarios, not confidence intervals or guaranteed bounds. Because all 300 tasks are used, a bootstrap sampling interval would not quantify the dominant uncertainty: missing call/observation/output logs.

## Input/output and task complexity

| Approach | Mean input | Mean output | Actor total | WM + verifier total | 1-app mean | 2-app mean | 3-app mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours (GPT-5.4 trajectories) | 31.5k | 3.0k | 19.2k | 15.3k | 20.6k | 26.4k | 53.1k |
| B1 (GPT-5.4 trajectories) | 58.0k | 3.4k | 23.5k | 38.0k | 40.1k | 44.1k | 93.9k |
| AgentWorld (Qwen3.5 trajectories) | 92.8k | 3.7k | 28.8k | 67.7k | 69.1k | 74.4k | 138.0k |
| Ours, raw context + persistent WM transcripts | 169.4k | 3.0k | 19.2k | 153.2k | 70.2k | 92.4k | 325.2k |
| Ours, raw context only | 44.9k | 3.0k | 19.2k | 28.7k | 27.7k | 34.9k | 75.8k |
| Ours (Qwen-AgentWorld trajectories) | 66.6k | 6.2k | 43.1k | 29.6k | 43.9k | 61.2k | 106.6k |

## Rollout scaling

Approximate independent-candidate budget: T(task, N) ≈ N × T(task, 1). These are planning extrapolations, not measurements of the saved N>1 cells. They exclude extra comparative reranking and assume no response-cache savings. For a reused nested pool N=1,2,4,8,16, generating the largest pool is about 16 times N=1, not 31 times, plus selection work.

| Approach | N=1/task | N=2/task | N=4/task | N=8/task | N=16/task | All 300, N=1 |
|---|---:|---:|---:|---:|---:|---:|
| Ours (GPT-5.4 trajectories) | 34.6k | 69.2k | 138.3k | 276.6k | 553.3k | 10.37M |
| B1 (GPT-5.4 trajectories) | 61.5k | 122.9k | 245.8k | 491.7k | 983.4k | 18.44M |
| AgentWorld (Qwen3.5 trajectories) | 96.5k | 193.0k | 385.9k | 771.8k | 1,543.7k | 28.94M |
| Ours, raw context + persistent WM transcripts | 172.4k | 344.9k | 689.7k | 1,379.4k | 2,758.9k | 51.73M |

## Method and assumptions

1. Loaded N=1 records from the four source runs in `provenance.json`. Every source token field is zero; these cannot be used as usage measurements. Parsed saved action sequences, retaining malformed steps and salvaging visibly truncated arguments. The GPT-5.4 ours/B1, AgentWorld, and Qwen ours records have respectively 3,228 / 3,627 / 4,220 / 5,375 saved steps; 100 / 244 / 479 / 695 action strings require partial reconstruction. Long missing document arguments can materially increase real usage.
2. Read pristine initial state, gold references, app catalogue, and task instructions for all 300 tasks using the existing OfficeBench driver. Built actor, B1 fixed-context, AgentWorld system, and verifier prompt text with current workspace functions. This reconstructs current prompt templates over historical trajectories; it does not recover the exact historical requests.
3. Counted text with `cl100k_base` as a common tokenizer proxy. Neither GPT-5.4 nor Qwen provider billing counts are claimed. Missing observations use matching pristine content, a file listing for shell actions, or a fallback source document. Writes use short status text. Actual post-write content, filter results, errors, OCR text, and malformed-action corrections can differ.
4. For ours/B1, modeled prediction calls from writes, reads of dirty state, and changed evaluator files without gold-backed reconstruction exemption. A base prediction prompt uses the actual simulator system and Python protocol plus an assumed 320-token request header and 150-token evidence/fact allowance, plus action tokens. Verifier per-check counts use saved breakdowns; absent breakdowns are treated as deterministic early rejection. A whole-task audit is added when checks ran. Final state supplied to verifier is approximated from pristine state; actual predicted state/write-log length was not logged.
5. Assumed low/central/high observation payload sizes of 0.5× / 1× / 2× reconstructed text; 1 / 1.5 / 2.5 calls per prediction; and 0 / 100 / 400 additional completion tokens per actor or generative-WM call. Recorded Python execution counts provide another call-count proxy via max(prediction calls, Python calls); Python calls are not equivalent to LLM calls and can include verifier work. Completion bases use visible action tokens, short status/program sizes, or reconstructed observation sizes. These assumptions are not fitted measurements of reasoning tokens or repair behavior.
6. AgentWorld sends its reconstructed system/snapshot plus accumulating action/response history for each valid simulated action, and one verification transition. Raw malformed actor outputs consume actor tokens but do not trigger a transition. Full returned reasoning content was not logged, so its contribution is scenario-based.
7. The saved full GPT-5.4 runs used the clean-read execution setting (predict_all_actions absent/false), Python WM mode, and up to 100 actor steps. AgentWorld used Qwen3.5-35B-A3B and a 40-step limit; Qwen ours used Qwen-AgentWorld-35B-A3B and 100 steps. Thus the approach comparison is not a controlled same-model experiment, and these numbers must not be presented as the fully action-free predict-all ablation.
8. Includes actor + modeled simulator + verifier language-model tokens. Excludes unlogged app-internal LLM calls, any LLM work inside final real replay, infrastructure failures, prefix-cache pricing effects, and N>1 comparative reranking. High scenarios are not upper bounds.

## Meaning of “without subagent context management”

The primary counterfactual holds actions, modeled call counts, and completions fixed, but includes raw pristine evidence and preceding actor history in each WM prompt and retains prior non-system WM request/response transcripts for later WM and verifier calls. In notation, the additional input is approximately Σ_j (raw_context_j + Σ_(i<j) retained_transcript_i). For fractional modeled calls, the script uses the corresponding arithmetic-series interpolation. Retaining transcripts restarts per candidate; it does not leak across candidates. This estimates token cost conditional on the same behavior, not the success rate or actual trajectory of a rerun.

The narrower raw-context-only counterfactual adds raw evidence and actor history to each WM request but preserves independent subagent boundaries. It costs about 48k/task (1.39× ours) centrally. Retaining all prior subagent transcripts as well costs about 172k/task (4.99× ours), with a much heavier tail. These are two distinct ablations; there is no uniquely defined “no management” cost without specifying which mechanisms are removed.

## Artifacts and reproducibility

- `estimate.py`: executable estimator; no API calls or candidate writes.
- `per_task.csv`: per-task/component estimates for all three scenarios.
- `summary.json`: means, median, SD, P10/P90/P95, maximum, subset summaries.
- `provenance.json`: source run paths, units, tokenizer, scenarios.
- `snapshots/`: cached pristine snapshots for all 300 tasks.

Run from the repository root: `PYTHONPATH=src:/tmp/officebench-token-deps python experiments/officebench-token-estimate-20260907/estimate.py`. The temporary dependency directory contains tiktoken and openai (the latter is needed only to import existing AgentWorld prompt helpers). A compatible environment with those packages can omit that directory. Existing cached snapshots avoid further Docker work.

## Qwen figure update — 2026-09-08

The figure now separates Qwen and GPT-5.4 in both panels. GPT-5.4 rows are
explicit placeholders; historical GPT estimates remain in the underlying files.
Qwen B1 is also pending (`b1_qwen: null`): the available `qwen_offic_b2` run is
labeled B2 and is not substituted for B1.

The Qwen context ablations are derived from the 300 existing `ours_qwen` per-task
rows, using their saved `raw_only_input` and `noctx_input` fields. They preserve
the baseline's actions, modeled calls, and output tokens. No model was called.

| Qwen variant | Central mean tokens/task | Relative to Ours |
|---|---:|---:|
| Ours | 72,757.9 | 1.00× |
| + raw evidence and actor history | 115,190.8 | 1.58× |
| + raw context and prior WM transcripts | 643,151.0 | 8.84× |

`complete_qwen_summary.py` reproducibly updates the Qwen ablations in
`per_task.csv`, `summary.json`, and provenance using only the saved estimates.
`estimate.py` now also emits both Qwen ablations when rebuilding estimates.
Run `python experiments/officebench-token-estimate-20260907/complete_qwen_summary.py`,
then `PYTHONPATH=/tmp/officebench-plot-deps python experiments/officebench-token-estimate-20260907/plot_token_usage.py`.
The plot exports PDF, SVG, PNG, CSV (pending values are blank), and a LaTeX caption.
Qwen denotes the model family: the saved Ours and AgentWorld runs use different
Qwen variants and step limits, as documented above.

## GPT-5.4 replacement — 2026-09-08 (collected AML trajectories)

The GPT-5.4 placeholders in `summary.jsonl` have been replaced with estimates
reconstructed from saved AML trajectories. The remaining new jobs were stopped
at the user's request. B1 uses a complete 300-task N=1 run; Ours uses 139 unique
completed tasks: 93 one-app, 31 two-app, and 15 three-app. All these GPT-5.4 runs
use temperature 1, the clean-read setting, Python WM mode, and a 40-step limit.
This differs from the historical trajectories and 100-step limit described above.

Ours' means are poststratified to the full benchmark's app-count proportions:

`mean = (93/300)*mean_1app + (95/300)*mean_2app + (112/300)*mean_3app`

Each observed task receives weight `population_subset_count / observed_subset_count`.
The summary's `n` fields report actual observations (139 for Ours and its ablations,
300 for B1); `population_n_tasks: 300` is the target population, not claimed coverage.
Weighted quantiles interpolate at cumulative-weight midpoints. This is a
completion-selected sample, not a random sample: unfinished expensive trajectories
may be underrepresented even after weighting. Low/high scenario whiskers do not
measure this sampling uncertainty.

| GPT-5.4 method | Observed tasks | Mean input | Mean output | Mean total | Low–high mean |
|---|---:|---:|---:|---:|---:|
| Ours | 139 | 31.3k | 2.8k | 34.1k | 28.5k–51.0k |
| B1 | 300 | 99.1k | 5.7k | 104.8k | 73.9k–175.2k |
| + raw evidence and actor history | 139 | 51.0k | 2.8k | 53.8k | 39.9k–96.5k |
| + raw context and prior WM transcripts | 139 | 316.9k | 2.8k | 319.7k | 215.8k–827.6k |

The two context ablations hold each observed Ours trajectory, modeled call count,
and completion budget fixed, adding only input context. Estimates use the
observation/call/completion scenarios documented above and `cl100k_base`; they are
not measured API usage. The reconstructed snapshot readers, malformed-action
handling, fallback observation choices, and sampling limitations are documented in
`experiments/officebench_gpt54_tokens_20260908/README.md`.

Reproducible inputs and outputs are in that directory: `ours_records.jsonl`,
`merged_run_manifest.json`, `per_task.csv`, `gpt54_summary.json`, `provenance.json`,
`catalogue.json`, and cached `snapshots/`. Source AML run names and SHA-256 hashes
are recorded in provenance. Duplicate task results use a fixed source priority,
never a choice based on success or token cost. Qwen values are unchanged, and
Qwen B1 remains pending. `draw_token_consumption.py` now plots the GPT-5.4 values
and labels the partial weighted coverage in its figure, CSV and LaTeX caption.
