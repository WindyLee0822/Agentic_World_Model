# OfficeBench token budget approximation — 2026-09-07

These are **offline scenario estimates, not measured API usage**. Use them for rough budgeting, not as an empirical token-efficiency result. No new benchmark/model calls were made.

Sources: `qwen_offic_ours/records.jsonl` and `qwen_offic_b2/records.jsonl`, both complete Qwen-AgentWorld-35B-A3B, temperature-1 runs: 300 tasks each (93 one-app, 95 two-app, 112 three-app), N=1/2/4/8/16. All 3,000 saved decision records have `tokens=0`. The harness records spawn-tree tokens, not the OfficeBench agent + predictor + verifier calls. The records contain actions and Python execution counts, but lack complete prompts, observations, completions, and per-call usage. Current source may differ from the source used for those runs.

The unit is **total input + output tokens per task at one specified N**, summing all N candidates, world-model work, and selection work. Repeated input counts each time it is sent. This is not peak context length, generated tokens alone, or cache-discounted billing. B2 uses official execution-based scoring without an LLM success verifier. Committing a selected action list does not require another solver rollout. App-internal models (e.g. OCR or an OfficeBench llm app), hidden reasoning, retries, and unlogged failed calls are not separately counted.

## Main estimate

Mean total tokens per task, rounded; K=1,000 and M=1,000,000:

| Candidates N | B2 | Ours | Ours without context management* |
|---:|---:|---:|---:|
| 1 | 76K | 187K | 702K |
| 2 | 161K | 382K | 1.30M |
| 4 | 416K | 726K | 2.33M |
| 8 | 747K | 1.44M | 4.44M |
| 16 | 1.44M | 2.84M | 8.63M |

*Counterfactual definition: preserve the same solver trajectories, WM call-count proxy, verifier, and outputs, but put raw context into WM prompts and retain internal WM exchanges across calls within each candidate. Reset between candidates. This assumes no behavior change and no context-window failure. It is not an executed ablation. The modeled growth is quadratic in internal call count, so the definition matters greatly. If only raw-data hiding is removed but subagent histories still reset, use the much smaller `ours_raw_only` scenario in `summary.json`.

For that raw-only interpretation, base mean tokens/task at N=1/2/4/8/16 are **230K / 463K / 879K / 1.73M / 3.42M**. At N=1 the two possible ablation definitions therefore span 230K–702K even before varying prompt-size assumptions.

## Variation across tasks and sensitivity to assumptions

At N=1, under the base assumptions:

| Method | Mean | Median | P90 | Low–high scenario mean |
|---|---:|---:|---:|---:|
| B2 | 76K | 41K | 234K | 37K–195K |
| Ours | 187K | 82K | 520K | 89K–461K |
| Ours without context management | 702K | 144K | 2.23M | 217K–2.01M |

These percentiles describe variation in modeled task costs, not uncertainty intervals around measured tokens. Low/high are deliberately broad input-size scenarios, not confidence bounds. At N=1, the base input/output split is approximately B2 75.6K/0.5K and ours 182.3K/4.9K. Input dominates because history is resent.

Measured action counts at N=1: ours mean 17.96, median 11, P90 40; B2 mean 14.01, median 11, P90 32.1. At N=16, mean actions per candidate are 17.10 and 14.53 respectively. Ours records 15.82 Python executions/task at N=1 and 217.19/task at N=16. Python executions are **not** the same as LLM calls. Long documents and shell commands contain ` | `; the parser decodes JSON rather than naively treating these as extra steps. Malformed, truncated action entries fall back to their available text.

The 300-task dataset budget under base assumptions is approximately N=1: B2 22.9M, ours 56.2M, no-context 210.6M; N=16: B2 431M, ours 852M, no-context 2.59B. Each N is treated as a standalone budget estimate; do not add the table rows to estimate the historical sweep, because candidate pools can be reused across N.

## Assumptions and calculation

| Assumed token quantity | Low | Base | High |
|---|---:|---:|---:|
| Solver system + task + current-app instructions per call | 1,200 | 2,000 | 3,500 |
| Observation per solver step | 100 | 300 | 1,000 |
| Managed WM input per call | 1,200 | 2,500 | 5,000 |
| WM output per call | 100 | 200 | 400 |
| Atomic/comparative verifier fixed input, excluding trace | 1,200 | 2,500 | 6,000 |
| Extra raw context per unmanaged WM call | 500 | 2,000 | 6,000 |
| Prior internal exchange retained per unmanaged WM call | 250 | 1,000 | 3,000 |

Action text is approximated at four characters/token, not with the Qwen tokenizer. Solver history uses displayed action arguments clipped to 200 characters, an assumed observation size, and 12 tokens formatting per entry. Existing solver history is already cumulative in both methods. The current source's fixed solver system literals alone are roughly 640 tokens by this heuristic; task/catalogue/current-app instructions add to that.

For a candidate with L actions:

`solver_input = sum_t(base_prompt + sum_previous(displayed_action_tokens + observation_tokens + 12))`

`solver_output = sum(action_JSON_characters / 4)`

WM call proxy per task: the larger of observed Python executions and (non-system action count + one state call/candidate + one whole-verdict call/candidate with recorded per-check metadata). Allocate this proxy across candidates in proportion to their action-based count. Add the recorded atomic-check count as direct verifier calls; use recorded comparative metadata for pairwise selection calls. Whole-verdict and state costs are included in the WM proxy. Missing per-check metadata is treated as no atomic checks, consistent with deterministic early rejection, though exceptions may also cause missing metadata.

This proxy intentionally budgets predictions conservatively: clean reads can bypass the WM, state reconstruction may need zero or several calls, and Python failures/multiple programs can break the one-execution/one-LLM-call relationship. Without per-call traces these cannot be resolved. The per-check branch calls each atomic check once plus one whole-task verdict; the saved `verifier_samples=5` is not a reason to multiply every atomic check by five.

Managed WM input/output use the per-call assumptions above. Atomic verifier input includes the reconstructed whole trajectory; comparative input includes two average candidate trajectories. The unmanaged increment for M internal calls in each candidate is:

`extra_input = M * raw_context + M * (M-1) / 2 * retained_per_call`

The raw-only alternative drops the quadratic term. Neither scenario adds another copy of solver history to solver calls. Atomic/comparative prompts already include trajectory evidence in the managed estimate; their costs are held constant in the ablation.

## Reproduce and inspect

Run from the repository root:

```bash
/usr/bin/python3 -S experiments/officebench_token_estimate_20260907/estimate.py
```

`summary.json` contains source SHA-256 hashes, assumptions, observed trajectory statistics, means, population standard deviations, medians, P10/P90, and totals for every N/scenario. `per_task.csv` contains the modeled input/output totals for every task, method, N, and scenario, with app-count subsets for further grouping. All 300 tasks and all N candidates are checked for presence.

Relevant implementation: `src/tts/env/officebench_agent.py` (solver history), `officebench_wm_python.py` (raw data kept in Python variables and bounded read loop), `officebench_wm.py` (atomic + whole verification), `real_officebench.py` (rollout, predicted state, comparison), and `src/tts/harness.py` (saved token counter).

To turn this into an empirical result, log prompt/completion usage around every solver, WM, and verifier call and rerun a matched subset with an explicit unmanaged-context implementation. The present figures should not be cited as measured savings.
