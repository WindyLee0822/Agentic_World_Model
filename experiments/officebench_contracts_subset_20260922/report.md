# Completed OfficeBench reliability subset — 2026-09-22

**The targeted repairs work, but the simulator is not reliable end to end.**

Job: `dw-officebench-gpt54-contracts-subset-n124-20260922` (Azure status: Completed). Six diagnostic tasks, two at each app-count level; N=1,2,4; 40-step budget; GPT-5.4; explicit predict-all mode. All 18 expected evaluation cells completed. Benchmark execution took 24:01, excluding environment startup. The frozen source is code asset `f0d6b120-d913-4296-bd9b-0d582efc7559/versions/1`; hashes are in `source_manifest.json`.

## Task success and confidence

The baseline is the same six tasks from the completed v2 diagnostic, `dw-officebench-gpt54-reliability-v2-smoke-n124-20260921`.

| N | Previous native-evaluator successes | Current successes | Current all-zero candidate pools | Current selected false positives | Current selected false negatives | Exhausted candidates |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1/6 | 2/6 | 5/6 | 0 | 1 | 0/6 |
| 2 | 1/6 | 3/6 | 5/6 | 0 | 2 | 0/12 |
| 4 | 1/6 | 2/6 | 5/6 | 0 | 1 | 0/24 |

The calendar-to-Excel task `2-15/0` now passes at all three N values, with confidence 1.0. Calendar event creation `1-1/0` passes but still receives confidence 0.0. The other four tasks all have zero confidence for every candidate.

**Important qualification:** the extra N=2 native-evaluator pass on `3-12/0` does not establish valid PDF generation. The selected trajectory calls Word `create_new_file` and `write_to_file` on `power_failure.pdf`, which uses DOCX serialization despite the extension. That task's PDF check only tests file existence. Thus the reported 3/6 is native-evaluator success, not three independently verified correct user outcomes.

## Observation fidelity

Across selected real replays, exact observations improved from **13/167 (7.8%)** to **90/136 (66.2%)**. Removing the legacy `OBSERVATION:` prefix before comparison raises the old result to 21/167 (12.6%); the new result remains 90/136. These are pooled selected traces with different action mixes, not paired identical-action trials or a full-benchmark estimate.

| Action | Exact / replayed |
|---|---:|
| Calendar create_event | 3/3 |
| Calendar list_events | 3/3 |
| Excel read_file | 6/6 |
| Excel set_cell | 39/39 |
| Excel create_new_file | 3/3 |
| Word create_new_file | 13/13 |
| Word write_to_file | 13/13 |
| Excel convert_to_pdf | 0/3 |
| Word convert_to_pdf | 0/12 |
| OCR recognize_file | 0/6 |
| Email send_email | 2/15 |
| Email list_emails | 0/5 |
| Shell command | 8/15 |

The first seven rows match **80/80**. Their improvement does not extend automatically to conversion, OCR, email, or shell behavior. Some remaining differences are wording/order; others reflect incorrect state or execution outcomes.

## Remaining failures

1. **Conversion state and runtime behavior.** Conversion predictions still sometimes claim that a document is missing after it was created and written earlier in the same trajectory. Other predictions announce successful conversion, but native replay reports `/usr/bin/oosplash: not found`. The wrapper's success sentence is followed by a runtime error. Conversion needs both current-state evidence and a faithful model of available runtime capabilities.
2. **Email state after writes.** A predicted `list_emails` response says there are no emails for Tom, while native replay lists a newly sent message. This is a substantive state error, not just output formatting.
3. **Unreliable selection among zero-confidence candidates.** On `3-12/0`, N=2 selects candidate 1, which passes the native evaluator; N=4 replaces it with candidate 2, which fails. All candidate confidence scores are zero. The comparative reranker claims candidate 2 uniquely created the PDF and sent email, despite the contrary replay evidence. The N=2 artifact-format qualification above still applies.
4. **Incomplete generated-program contracts outside repaired actions.** The run logged four rejected `Starred` programs, three OCR string-as-dictionary errors, and two rejected `Delete` programs. Local repair prevented exhausted rollouts, but successful execution still does not guarantee correct state reconstruction.

The most useful next work is to extend consistent logical-state handling and native contracts to conversion/email, repair the native conversion runtime, and ground comparative selection in those states. Increasing N alone did not improve reliability monotonically.

## Paradigm and artifacts

Recorded simulator writes: **0**. Recorded real commits: **18**, exactly one per evaluation cell. No candidate trajectory was executed to obtain its predicted observations.

- `analysis/metrics.json`: matched-task comparison and per-action fidelity.
- `analysis/run_manifest.json`: complete coverage, with no missing cells or abandoned tasks.
- `analysis/failure_examples.json`: concrete conversion/email mismatches and the N=2 → N=4 selection regression.
- `analysis/records.jsonl`, `analysis/run.log`: complete downloaded evidence.
- `analysis/fidelity_comparison.json`: raw and prefix-normalized comparison.
- `tasks.json`: task instructions and native evaluator definitions.

This six-task diagnostic demonstrates remaining failures; it is not a statistically representative benchmark estimate.
