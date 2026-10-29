# OfficeBench repair validation — 2026-09-23

The repairs substantially improved simulator reliability on the targeted subset. They did not uniformly improve task completion: native success fell at n=1 and n=2, and rose at n=4. This small regression subset does not establish whole-dataset reliability.

Completed job: `dw-officebench-gpt54-evidence-r2-subset-n124-20260923`. Eight tasks at n=1,2,4, GPT-5.4, 24/24 expected records, unchanged native evaluator and fixtures. Baseline: matching tasks from `dw-officebench-gpt54-repaired-full-n124-20260922`.

## Results

| Metric | Previous version | Repaired version |
|---|---:|---:|
| Selected success/failure verdict agrees with native evaluation | 4/24 (16.7%) | 24/24 (100%) |
| False positive verdicts | 3 | 0 |
| False negative verdicts | 17 | 0 |
| Exact committed observation matches | 120/162 (74.1%) | 139/141 (98.6%) |
| Recorded Python failures / calls | 76/678 (11.2%) | 3/235 (1.3%) |
| All-zero candidate pools | 21/24 | 8/24 |
| Simulator environment writes | 0 | 0 |
| Selected trajectory commits | 24 | 24 |

| n | Native successes, before → after | Correct verdicts, before → after |
|---|---:|---:|
| 1 | 6/8 → 5/8 | 1/8 → 8/8 |
| 2 | 6/8 → 5/8 | 1/8 → 8/8 |
| 4 | 5/8 → 6/8 | 2/8 → 8/8 |

The observation denominators differ because the sampled trajectories changed. These are comparisons over the same task set, not identical action sequences. The three new Python failure-counter events all occur in task 3-12/0, one in each n record. No corresponding detailed `officebench-python-error` entries were emitted; do not infer a specific exception from the counters. Candidate reuse across n means these are not necessarily independent failures.

## Repairs validated

- **Evidence routing:** email checks receive the predicted mailbox; existence-only document checks use the right file type. Task 3-12/0 now has correct positive verdicts at every n, replacing false negatives.
- **Check semantics:** supported native predicates run as pure functions over predicted state. They handle dictionary-shaped cell checks, paired-file diffs, whole-predicate negation, and conservative exact matching. This removes repeated model-generated verifier programs for those checks. Tasks 1-4/0 and 1-8/0 now have correct positive verdicts at every n.
- **State materialization:** supported edits to files with gold references produce predicted resulting content rather than an initial-content/write-log description. Bounded move/copy and directory-creation transitions maintain copied state. Task 1-3/0 now has correct positive verdicts at every n.
- **Observation contracts:** single-email reads use the native full-body format and suffix. Exact email-read agreement changed from 0/9 to 10/10. Shell observation agreement changed from 25/38 to 26/26 on the committed action mixture. Repeated-action hints no longer contaminate application observations.
- **Selection:** an all-zero pool cannot discard the incumbent merely because of speculative invalidity; a challenger needs positive audited progress. The previous n=4 failure on 3-65/0 is now a native success. Since observations and verdicts also changed, this is not an isolated causal test of the selection rule.
- **Native runtime capability:** the simulator can predict the missing LLM credential-file error using a presence flag without reading credentials. It matched 2/4 committed LLM app calls; the remaining two were quoting failures.

All candidates continue to have their observations predicted before selection. Only the selected trajectory is executed in the native environment. Deterministic helpers operate on copied inputs and predicted state; they do not execute candidate actions in the environment. Unsupported checks retain the previous verifier path.

## Why native success was mixed

Task **1-20/0** regressed at n=1 and n=2. The actor answered without the required `Subject: scheduled meeting` email and included a January email in its recent-email answer. The simulator correctly marked the required positive containment check UNMET. At n=2 both candidates failed; the incumbent was retained. At n=4 candidates 2 and 3 passed the predicted checks, candidate 2 was selected, and native evaluation passed. These are actor/trajectory failures that the new simulator detects, not incorrect positive simulator verdicts.

Task **3-65/0** succeeded at every n, recovering the previous n=4 native failure. Consequently overall native successes were 17/24 before and 16/24 after, despite substantially better prediction accuracy.

Two negative controls remain failures at every n:

- **1-7/0:** the diff evaluator references an unprovisioned cache input. The simulator correctly predicts failure; the environment was not silently modified.
- **3-82/0:** the evaluator requests XLSX decoding for a DOCX filename. All three previous false positives are now correct negative verdicts.

## Remaining gaps

1. **LLM app command-line quoting:** both remaining observation mismatches occur on 3-65/0 when apostrophes in the prompt cause native CLI parsing errors. The simulator returns `PREDICT:` instead. Missing-key handling alone does not cover malformed CLI invocations.
2. **Actor completeness:** accurate observations and scoring do not guarantee a good candidate exists, as 1-20/0 at n=1,2 demonstrates.
3. **Generalization:** arbitrary shell effects, dirty directory listings, calendar reconstruction, and some document-conversion contracts remain outside the bounded pure helpers. Numeric XLSX exact checks remain UNKNOWN where decoding loses cell types. Correct predicates cannot compensate for incorrect predicted state.
4. **Diagnostic coverage:** Python output/contract failure counters should carry reason/context logs as execution exceptions already do. The three remaining counter events cannot be precisely diagnosed from this run's logs.

The next focused validation should cover malformed LLM prompts and a broader held-out set spanning dirty state and conversions. This run supports retaining the repairs, but is not evidence that every simulator path is reliable.

## Validation and reproducibility

Local regression suite: **183 passed, 6 skipped**. Runtime preflight passed, including conversion and network isolation.

- `comparison.json`: per-n counters, paired task records, and observation counts.
- `records.jsonl`, `run.log`, `observation_mismatches.json`: raw evidence.
- `regression_tests.log`: local test result.
- `../analyze_subset.py`: reproducible paired comparison.
- `../changes.patch`, `../test_changes.patch`, `../source_manifest.json`: changes and submitted source hashes.
- `../job.yml`, `../completed_job.json`: submitted configuration and completed status.

Production files were verified byte-for-byte against the submitted snapshot. The native evaluator and dataset were unchanged. Only selected trajectories have a native observation oracle; the unselected candidates were not executed to establish their ground-truth outcomes.
