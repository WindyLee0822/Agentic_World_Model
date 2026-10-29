# Runtime and state repair: four-task subset

Job: `dw-officebench-gpt54-final-state-n124-20260922`. Four conversion/email tasks, N=1,2,4, GPT-5.4, predict-all, select then commit one candidate.

## Native task results

| N | Before | Repaired | All-zero pools | Exhausted candidates |
|---|---|---|---|---|
| 1 | 2/4 | 3/4 | 2/4 | 0/4 |
| 2 | 2/4 | 3/4 | 2/4 | 0/8 |
| 4 | 2/4 | 3/4 | 2/4 | 0/16 |

## Success-prediction agreement

| N | Correct / tasks | False positives | False negatives |
|---|---|---|---|
| 1 | 3/4 | 0 | 1 |
| 2 | 3/4 | 0 | 1 |
| 4 | 3/4 | 0 | 1 |

Exact selected-action observation fidelity: **117/117 (100.0%)**. Action mixtures differ across runs; these are not paired identical-action trials.

| Action | Exact / total |
|---|---|
| calendar.list_events | 3/3 |
| email.send_email | 12/12 |
| excel.convert_to_pdf | 3/3 |
| excel.create_new_file | 3/3 |
| excel.read_file | 3/3 |
| excel.set_cell | 60/60 |
| ocr.recognize_file | 6/6 |
| shell.command | 9/9 |
| word.convert_to_pdf | 6/6 |
| word.create_new_file | 6/6 |
| word.write_to_file | 6/6 |

Simulator writes: 0; real commits: 12.

## Task outcomes

| Task | N | Success | Confidence | Selected |
|---|---|---|---|---|
| 2-1/0 | 1 | True | 0.8 | 0 |
| 2-1/0 | 2 | True | 0.8 | 0 |
| 2-1/0 | 4 | True | 1.0 | 2 |
| 2-15/0 | 1 | True | 1.0 | 0 |
| 2-15/0 | 2 | True | 1.0 | 0 |
| 2-15/0 | 4 | True | 1.0 | 0 |
| 3-1/0 | 1 | False | 0.0 | 0 |
| 3-1/0 | 2 | False | 0.0 | 0 |
| 3-1/0 | 4 | False | 0.0 | 0 |
| 3-12/0 | 1 | True | 0.0 | 0 |
| 3-12/0 | 2 | True | 0.0 | 0 |
| 3-12/0 | 4 | True | 0.0 | 0 |

Native evaluator scores are reported unchanged. A passing file-existence check alone does not establish PDF validity. The runtime preflight independently validates synthetic Word/Excel PDFs, text content, shell listing, and network denial. Per-check selection audits remain predictions.

See `README.md`, `changes.patch`, `tests_main.txt`, `tests_other.txt`, `analysis/metrics.json`, and `analysis/observation_mismatches.json` for implementation and evidence.

Task `3-1/0` has an independently checked evaluator/fixture mismatch: its checks require `Activaty1..4`, whereas the spreadsheet contains `Activity 1..4`. See `evaluator_fixture_mismatch.json`. No evaluator or fixture change was made.

## Subsequent verifier repair: separate offline audit

After the completed job, the verifier was repaired to accept successful recovery, forecast the
native-check conjunction, and inspect private task/state/gold values only through copied Python
input. Email checks explicitly resolve `emails/<username>` rather than expecting a file argument.
The job tables above remain the original recorded results; this follow-up did not rerun selection
or execute native actions.

| Frozen trajectory / control | Repaired confidence | Atomic checks |
|---|---:|---|
| Successful 3-12/0 trajectory after a failed combined-recipient send | 0.8 (previously 0.0) | 5 MET |
| Same trajectory with wrong Tom mailbox content | 0.0 | 3 MET, 2 UNMET |
| Same trajectory with Tom mailbox evidence omitted | 0.0 | 3 MET, 2 UNKNOWN |

All three live GPT-5.4 audits passed their expectations: 25 Python calls, zero Python failures,
zero native actions, and private-content prompt checks passed. Whole-task voting accepted the
successful case in 4/5 votes, so some scoring variance remains. This four-task subset and the
three scorer controls do not establish reliability on all 1,248 tasks or uncommitted candidates.

Regression suite: 177 passed, 6 skipped. After the final email-routing instruction change,
13 focused verifier tests passed. Evidence: `verifier_followup.json`, `verifier_followup.log`,
`verifier_followup_programs.jsonl`, `verifier_followup.patch`, and the follow-up test logs.
The two earlier unsuccessful scorer probes are retained as `verifier_followup_attempt1.*`
and `verifier_followup_attempt2.*`.
