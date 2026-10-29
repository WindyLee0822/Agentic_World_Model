# GPT-5.4 token estimates from AML runs

These are reconstructed token estimates under `token_assumption.md`, not measured
API usage. B1 includes all 300 tasks (including failures) at N=1. Ours uses the
already-collected, completed trajectories after the user requested an approximation
without waiting for full new runs. Raw-context
ablations reuse Ours' actions, modeled calls, and completions.

Sources in `external-collaborations/M365Research-External-EUS`:

- B1: `dw-officebench-b1-llm-wm-gpt54-n124-20260904`; select only its N=1 records.
- Ours one-app: `dw-officebench-ours-gpt54-tokens-n1-20260908`; retain its complete
  93-task one-app group plus its already-completed two-app trajectories.
- Additional collected two-app trajectories: `dw-officebench-ours-gpt54-tokens-n1-2app-20260908`.
- Collected three-app trajectories: `dw-officebench-ours-gpt54-tokens-n1-3app-20260908`.
  These jobs were stopped at the user's request; only completed records are used.
  All three new N=1 runs use B1's saved code asset, environment, GPT-5.4 deployment,
  temperature 1, clean-read setting, Python WM, 40-step limit, and task seed policy.
- Task files and app documentation: code asset
  `328cf242-2c2e-43f9-867e-b99b9ff78638/1`, packaged with the B1 job.

Ours is poststratified to the benchmark's app-count proportions. Each observed
one-/two-/three-app task receives weight `93/n1`, `95/n2`, or `112/n3` respectively;
the weighted mean divides the weighted total by 300. The summary's `n` fields
remain the actual observed sample size. Weighted percentiles interpolate at the
midpoints of cumulative weights; full-coverage B1 uses ordinary linear quantiles.
Completion-selected partial runs are not random samples: weighting corrects app
proportions but cannot recover slow unfinished trajectories or eliminate selection
bias. The low/high scenarios do not quantify that sampling uncertainty. The collected
Ours sample is 139 tasks: 93 one-app, 31 two-app, and 15 three-app.

`merge_runs.py` keeps one trajectory per task with fixed source priority: original
run, then two-app run, then three-app run. It never chooses a duplicate by token
cost or success. `merged_run_manifest.json` records coverage, skipped duplicates,
and missing task IDs.

`estimate.py` reconstructs actor prompts with the repository's prompt helpers.
It reads pristine OfficeBench documents without executing saved actions, and
caches their text and the official app catalogue. Its three scenarios use
observation multipliers 0.5/1/2, prediction-call multipliers 1/1.5/2.5, and extra
completion allowances 0/100/400 tokens. The common tokenizer is `cl100k_base`.

Prediction work comes from writes, reads of dirty state, and changed evaluator
files. Saved Python executions provide a call-count proxy. Saved verifier
breakdowns supply atomic-check counts and whole-task audits. B1 repeats its
reconstructed fixed context. The raw-only counterfactual adds pristine evidence
and preceding actor history to WM/verifier requests; the persistent-transcript
counterfactual additionally retains previous non-system requests and responses,
with arithmetic-series interpolation for fractional modeled calls.

This is a newly implemented reconstruction of the documented assumptions, not
recovery of the missing historical estimator. Shell observations use pristine
file listings. Missing read operands fall back to the largest pristine source
document. Saved truncated arguments are partially reconstructed, and actual
post-write observations, internal repair prompts, and reasoning are not recovered.
Low/high values are sensitivity scenarios, not confidence intervals. The
`fixed_prompt` component reports the WM system-prompt size (including B1 fixed context).

The auxiliary `token_calls.jsonl` in the new AML output is not used for these
estimates: shared append logging can interleave across worker processes. The
harness's `tokens=0` fields are likewise not treated as API measurements.

Artifacts:

- `gpt54_summary.json`: the four replacement GPT-5.4 blocks.
- `per_task.csv`: modeled per-task input/output and component costs by scenario.
- `provenance.json`: source run names, file hashes, assumptions and limitations.
- `catalogue.json`, `snapshots/`: cached pristine reconstruction inputs.
- `b1_run/`, `ours_run/`, `ours_run_2app/`, `ours_run_3app/`: saved AML outputs.
- `ours_records.jsonl`, `merged_run_manifest.json`: collected matched Ours records with
  per-record source jobs; duplicate work from the original run is excluded.
- `aml_run*.json`, `instrument_run*.py`: new-job metadata and instrumentation.

Reproduce with tiktoken, python-docx, openpyxl, PyMuPDF and pytesseract installed
(Tesseract is needed only when rebuilding image snapshots):

```bash
python experiments/officebench_gpt54_tokens_20260908/merge_runs.py
PYTHONPATH=src:/tmp/officebench-token-deps python \
  experiments/officebench_gpt54_tokens_20260908/estimate.py \
  --ours-records experiments/officebench_gpt54_tokens_20260908/ours_records.jsonl \
  --update-summary
python draw_token_consumption.py
```

Cached snapshots and catalogue avoid needing the original task/code download.
To rebuild those caches, pass `--officebench-root` pointing to `officebench_data`
from the source code asset. The replacement preserves every non-GPT summary block.
