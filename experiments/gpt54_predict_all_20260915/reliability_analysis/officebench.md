# OfficeBench predict-all reliability audit

Run: `dw-officebench-ours-llm-wm-gpt54-predict-all-n1248-20260915`

Audited 2026-09-18. Azure ML reports Completed. All 1,200 expected task/N records are present; recorded rollout errors are zero. Evidence: downloaded `records.jsonl`, `summary.txt`, `run.log`, and the local submission staging tree `/tmp/gpt54-predict-all-20260915`. Download directory: `/tmp/officebench_reliability_20260918`. Source references below are relative to the staging tree, which matches the corresponding local source inspected in this audit; the remote code asset was not independently downloaded.

## Measured outcome

Each N covers the same 300 tasks, with shared candidate prefixes. Do not add counts across N as independent samples.

| N | Official successes | Selected predicted successes | True positives | False positives | False negatives | All candidate confidences zero |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 22 (7.33%) | 7 | 4 | 3 | 18 | 281 |
| 2 | 24 (8.00%) | 7 | 4 | 3 | 20 | 280 |
| 4 | 36 (12.00%) | 9 | 6 | 3 | 30 | 275 |
| 8 | 46 (15.33%) | 10 | 7 | 3 | 39 | 273 |

N=8 details:

- Success by difficulty: 25/93 one-app, 15/95 two-app, 6/112 three-app.
- Success-prediction recall: 7/46 = 15.2%; precision: 7/10 = 70% (small positive sample).
- 36 successful selected trajectories have confidence exactly zero.
- 217/300 selected trajectories contain at least 40 recorded actions; 156/300 contain both at least 40 actions and at least 30 shell commands. These are counts of serialized actions, not an assertion that every such trajectory hit the productive-step limit.
- Across the 2,400 candidates: 75,786 counted Python calls and 7,478 counted Python failures (9.87% ratio). Failures can be repaired, and repair counters are incomplete; this is not the fraction of actions with an unrecovered failure.
- 186/300 selected trajectories have at least one counted Python failure.
- 1,875/2,400 candidates have no per-check breakdown, consistent with the deterministic precheck returning early. Percheck is enabled and benchmark tasks have evaluation requirements, but the exact precheck failure reason was not retained in records.
- Of 525 available per-check breakdowns, 517 have whole-task score zero.

## 1. Reproduced bug: internal resource keys are reused as filesystem paths

`officebench_agent.py:344` starts the working directory at `/`.

`officebench_wm_predict.py:920` (`_shell_read_target`) normalizes a shell operand into a key relative to `/testbed`. For `find /testbed/data -type f`, the returned key is `data`.

`officebench_wm_predict.py:2273` then passes that key to `query.list(target, recursive=True)`. `OfficeBenchWMQuery.list` resolves it against the current working directory, rather than the testbed root. At the default `/`, `_testbed_path` rejects it:

```
cwd=/             target=data -> ERROR: LIST may access only directories under /testbed
cwd=/testbed      target=data -> /testbed/data
cwd=/testbed/data target=data -> /testbed/data/data
```

A local fake-session reproduction verified these results without an LLM or executing candidate actions. The valid absolute operand has been lost before the immutable listing executes.

The read predictor then fails to JSON-decode the error, substitutes `immutable_objects=[]`, and puts the error in a separate context field. Its Python protocol tells the model that this inventory is complete, to return immediately, and not to request another LIST. Thus a grounding failure can become an empty, apparently successful observation. A capture of the actual predictor context confirmed `target='data'`, `immutable_objects=[]`, and the scope error for the standard find command.

This provides a direct mechanism for the repeated search loops. Example: selected task `1-7/1` repeatedly searches for score spreadsheets throughout the action budget. The record contains no native-vs-predicted observation pairs, so the exact per-task causal contribution is not measurable from the archive.

Fix priority: convert canonical resource keys to absolute `/testbed/...` paths before query operations; preserve an explicit error/unknown state when listing fails. Never interpret a failed listing as an empty directory. Verify root and nested working directories.

## 2. Python prediction failures become fictitious application failures

`officebench_wm_predict.py:2089` allocates only 120 output tokens to write prediction, including its read-query programs, final programs, and repair programs. That budget originated from a single-line observation but is now shared with mandatory generated Python.

`officebench_wm_python.py:550` hides evidence values from model turns, exposing variable names; generated Python must inspect them. The predictor must therefore express app preconditions, history inspection, and output construction within this budget. Token pressure is a plausible contributor, but raw generated programs and finish reasons were not archived, so truncation frequency cannot be established.

The confirmed semantic problem is at `officebench_wm_predict.py:2104`: an invalid prediction becomes `World model failed to predict safely; the action was not applied.` It is treated as a failed write and omitted from the applied state. Simulator inability is therefore presented as environmental failure, prompting retries and leaving the verifier with missing/unchanged artifacts.

Run evidence: task `1-6/2` explicitly asks the LLM app for help because `excel.set_cell` repeatedly returned `World model failed to predict safely`. Calendar task `1-1/2` has eight zero-confidence candidates, yet the selected trajectory succeeds in real execution after repeated create_event calls. These establish disagreement and user-agent exposure to simulator errors, without identifying every underlying Python exception.

Fix priority: separate simulation-error/unknown from app failure; give program generation its own adequate budget; log failed code, error, finish reason, and repaired/unrecovered status.

## 3. Success confidence collapses and has the wrong interpretation

The run really used per-check verification, unknown credit 0.25, whole-task weight 0.25, comparative reranking, and nominal verifier_samples=5 (recorded metadata, not defaults).

`officebench_wm.py:410` applies a deterministic precheck to PREDICTED final state. Missing predicted output can cause immediate zero even if the real actions would create it. This amplifies simulator failures into ranking failures. Absence of per-check breakdowns in 1,875 candidates is consistent with this path.

For candidates reaching the judge, confidence is `0.75 * atomic_score + 0.25 * whole_score`. Partial requirement satisfaction is not the probability of passing every official check. Example `2-45/0`: atomic score=1, whole score=0, confidence=0.75, official failure. Example `1-7/1`: atomic score=0.75, whole score=0, confidence=0.5625, official failure. The latter reranker even notes there are no actual write actions while choosing a predicted output containing evaluator keywords.

Although records say verifier_samples=5, the active per-check branch explicitly calls `_predict_whole(samples=1)` at `officebench_wm.py:440`; the atomic statuses are also single calls. It does not deliver five whole-task votes in this configuration.

On 91% of N=8 tasks, all candidates tie at zero, leaving comparative reranking to choose among them. More candidates can still help (observed 7.33% to 15.33%), but the main confidence signal barely distinguishes success.

Fix priority: do not call uncertain simulated state a proof of failure; keep unknown separate, and distinguish partial-progress score from all-check pass probability. Measure calibration against real execution after repairing observation grounding.

## 4. State fidelity has additional structural limits

- `officebench_driver.py:247` truncates file reads to 2,000 characters, including requested source and reference files. Calendar/email extras are capped at 1,500. The predictor's nominal 6,000-character and verifier's 4,000-character limits cannot restore discarded content.
- Under program_only, `predict_read` and `predict_file_state` skip their deterministic symbolic fast paths (`officebench_wm_predict.py:2186,2380`). Python READ returns pristine content (`officebench_wm_python.py:370`), so generated programs must reconstruct relevant transitions. This increases the opportunities for drift on multi-app tasks.
- Native app semantics are not fully specified to observation prediction: its write prompt has app/action, footprint, and copied history rather than the app implementation. Successful execution of generated Python does not establish faithful emulation of the OfficeBench app.
- `real_officebench.py:618` replays the selected action sequence without replanning from actual observations. An incorrect simulated read can therefore determine all subsequent actions, and real execution cannot recover.

These are code-level risk mechanisms; the archive does not permit exact failure attribution to each one. The three-app success rate of 5.36% is consistent with compounded errors, but task difficulty is a confounder.

## Recommended order

1. Fix and verify the canonical-path/listing-error bug.
2. Separate simulator failure from application failure; improve program budgets and error observability.
3. Preserve complete required evidence and deterministic transitions wherever the experimental contract permits.
4. Correct the verifier's unknown handling, confidence semantics, and sampling configuration.
5. Run paired native/simulated action checks on a small representative set, logging observations, state differences, Python programs/errors, and precheck reasons before another full sweep.

No production code was changed. Unselected candidates were not officially executed in this run (`candidate_grounded_success` is null), so candidate oracle pass@8 and the amount lost exclusively to selection cannot be computed from these records.
