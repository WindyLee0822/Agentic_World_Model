# Runtime and state repair: six-task subset

Job: `dw-officebench-gpt54-runtime-state-r2-n124-20260922`. Same six tasks, N=1,2,4, GPT-5.4, predict-all, select then commit one candidate.

## Native task results

| N | Before | Repaired | All-zero pools | Exhausted candidates |
|---|---|---|---|---|
| 1 | 2/6 | 3/6 | 5/6 | 0/6 |
| 2 | 3/6 | 3/6 | 5/6 | 0/12 |
| 4 | 2/6 | 3/6 | 5/6 | 0/24 |

Exact selected-action observation fidelity: **93/99 (93.9%)**. Action mixtures differ across runs; these are not paired identical-action trials.

| Action | Exact / total |
|---|---|
| calendar.create_event | 3/3 |
| calendar.list_events | 3/3 |
| email.send_email | 3/6 |
| excel.convert_to_pdf | 3/3 |
| excel.create_new_file | 3/3 |
| excel.read_file | 6/6 |
| excel.set_cell | 39/39 |
| ocr.recognize_file | 6/6 |
| shell.command | 9/12 |
| word.convert_to_pdf | 6/6 |
| word.create_new_file | 6/6 |
| word.write_to_file | 6/6 |

Simulator writes: 0; real commits: 18.

## Task outcomes

| Task | N | Success | Confidence | Selected |
|---|---|---|---|---|
| 1-1/0 | 1 | True | 0.0 | 0 |
| 1-1/0 | 2 | True | 0.0 | 0 |
| 1-1/0 | 4 | True | 0.0 | 0 |
| 1-5/1 | 1 | False | 0.0 | 0 |
| 1-5/1 | 2 | False | 0.0 | 0 |
| 1-5/1 | 4 | False | 0.0 | 0 |
| 2-1/0 | 1 | True | 0.0 | 0 |
| 2-1/0 | 2 | True | 0.0 | 0 |
| 2-1/0 | 4 | True | 0.0 | 0 |
| 2-15/0 | 1 | True | 1.0 | 0 |
| 2-15/0 | 2 | True | 1.0 | 0 |
| 2-15/0 | 4 | True | 1.0 | 0 |
| 3-1/0 | 1 | False | 0.0 | 0 |
| 3-1/0 | 2 | False | 0.0 | 0 |
| 3-1/0 | 4 | False | 0.0 | 0 |
| 3-12/0 | 1 | False | 0.0 | 0 |
| 3-12/0 | 2 | False | 0.0 | 0 |
| 3-12/0 | 4 | False | 0.0 | 0 |

Native evaluator scores are reported unchanged. A passing file-existence check alone does not establish PDF validity. The runtime preflight independently validates synthetic Word/Excel PDFs, text content, shell listing, and network denial. Per-check selection audits remain predictions.

See `README.md`, `changes.patch`, `tests_main.txt`, `tests_native.txt`, `analysis/metrics.json`, and `analysis/observation_mismatches.json` for implementation and evidence.
