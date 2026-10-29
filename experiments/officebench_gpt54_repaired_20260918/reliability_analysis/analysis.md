# Repaired OfficeBench run: remaining reliability failures

Run: `dw-officebench-ours-llm-wm-gpt54-repaired-n124-20260918`. Audited 2026-09-21.

Azure reports Completed; all 900 expected task/N records are present. The remote code asset is `466cdecf-8dc8-492b-86d3-070f712799e5/versions/1`. Every source file in the preserved submission staging tree matches the saved SHA-256 manifest. Run artifacts were downloaded to `/tmp/officebench_repaired_audit_20260921`; the comparison run is under `/tmp/officebench_reliability_20260918`.

## Results: improved, but still unreliable

| N | Previous successes / 300 | Repaired successes / 300 | All candidate confidences zero | Exhausted-error candidates | Tasks with all candidates errored |
|---|---:|---:|---:|---:|---:|
| 1 | 22 (7.33%) | 59 (19.67%) | 242 | 91 / 300 | 91 |
| 2 | 24 (8.00%) | 74 (24.67%) | 213 | 166 / 600 | 62 |
| 4 | 36 (12.00%) | 92 (30.67%) | 196 | 327 / 1200 | 42 |

The same tasks recur across N, so do not sum rows as independent observations. At N=4, selected trajectories with at least 30 shell commands fell from 180 in the old run to 13 in the repaired run (computed from serialized action strings; not an exact observation-fidelity metric). Three-app tasks still achieve only 19/112 = 16.96%, versus 34/93 one-app and 39/95 two-app.

## 1. Dominant failure: generated read programs and the Python tool disagree about the supported interface

Of 327 exhausted candidate errors at N=4:

| Error category | Candidates |
|---|---:|
| Disallowed imports | 266 |
| Disallowed methods / builtins | 44 |
| JSON output exceeds 20,000 characters | 10 |
| Grounding/path errors | 5 |
| Syntax error or index error | 2 |

310/327 (94.8%) are unsupported Python operations. 313 errors occur during `clean_read`, 9 during `final_file_state`, and 5 during grounding. There are no exhausted `write_observation` errors in this final N=4 pool.

Most frequent exact error suffixes:

- 129: `import from 'xml.etree' is not allowed`
- 85: `import of 'io' is not allowed`
- 22: `import of 'xml.etree.ElementTree' is not allowed`
- 13: `import of 'zipfile' is not allowed`
- 12 each: `method 'encode' is not allowed`, `call to 'bytes' is not allowed`
- 11: `import of 'zlib' is not allowed`
- 13 across read/state prediction: `method 'start' is not allowed`

The evidence contract is underspecified. `officebench_driver.py:_read_one_file` already renders XLSX into cell text, DOCX into paragraphs, PDF into text, and images into OCR text. Yet `officebench_wm_python.py` tells the generator it receives copied read results without a precise format/schema for each result. The model sees variable names and query paths, not the result values. The binary/XML decoding errors strongly suggest it is trying to parse native office files even though the host has already decoded them. Raw generated programs are not archived, so this last causal interpretation is an inference; the unsupported-operation errors themselves are recorded facts.

The advertised subset is also incomplete: the prompt recommends `re.finditer`, but the tool does not allow the normal regex-match `.start()` method. This explains the mechanism behind that recorded failure without needing any binary decoding.

Priority: define typed read-result schemas and format-specific pure helpers; align the documented language with the validator. Broadly enabling filesystem/archive access would not resolve the fact that these inputs are already rendered text.

## 2. Error handling now exposes the failures, but retries do not repair their cause

The previous repair correctly stopped reporting simulator errors as native application failures. However, a persistent error now discards the entire trajectory after three rollout attempts. At N=4, 42 tasks have no usable candidate; 124 tasks have at least one error. Effective usable pool sizes are: 4 candidates on 176 tasks, 3 on 28, 2 on 31, 1 on 23, and 0 on 42.

These are often reproducible interface errors, not transient service failures. For example, every candidate for `1-1/2` ends in the same final-state `.start()` rejection. The harness retries the same rollout seed; the predictor uses temperature zero, and `chat_client.py` caches identical temperature-zero prompts by default. An identical request can therefore replay the same bad completion. The cache mechanism is confirmed in source, but no cache-hit trace was saved for individual failures.

The Python failure counters on surviving candidates omit discarded attempts: failed candidates are replaced with empty trajectories and reward metadata without those counters. Thus the surviving 1,119 failures / 10,204 calls is not an overall simulator failure rate.

Priority: repair the local failed computation with explicit schema/capability information, retain its trace, and distinguish persistent unsupported operations from transient failures before retrying whole trajectories.

## 3. Deterministic state reconstruction still differs from native app semantics

Two selected high-confidence failures were reproduced using unchanged native function bodies on temporary files. See `reproduce_semantics.py` and `semantic_reproduction.json`. These checks do not modify benchmark data or execute a new full benchmark.

### Task 1-5/1: setting Excel text "1" does not yield cell value 1

The task asks to halve the apple quantity; the selected action sets cell B2 to `"1"`. Confidence is 1.0, official success is false.

Fire parses the command-line argument `1` as an integer. The native `excel_set_cell` implementation contains `if text == True: text = ''`. Because `1 == True`, the saved cell becomes blank (read back as None). The symbolic simulator simply stores `str(action['text'])`, producing `(2, 2): 1`.

The local reproduction confirms both results. The problem is a benchmark implementation quirk that the simulator must reproduce, even though its behavior is surprising.

### Task 2-34/0: Word writes a DOCX container to a .txt path

The selected trajectory calls Word create/write on `/testbed/data/lowest.txt` with `16.91`. Confidence is 1.0, official success is false. The evaluator reads that file as plain text.

The Word app always uses python-docx, irrespective of the filename extension. The produced file starts with ZIP magic `504b0304`; its DOCX paragraph is `16.91`, but reading it as plain text does not contain the expected value. The symbolic simulator returns the plain string `16.91` regardless of the target format. Task `3-44/0` likewise uses Word to create `daily_schedule.txt`, receives confidence 1.0, and fails.

Other source-level mismatches remain: Word's command constructor strips quote characters while symbolic writes preserve them; dirty-read shortcuts return content without all native observation wrappers; spreadsheet rendering omits some native whitespace/empty-cell behavior. These mechanisms are visible in source, but their run-wide contribution is not quantified.

Priority: represent file format and CLI argument conversion, not just a path-to-text map. Validate supported symbolic transitions against the actual app wrappers on isolated files. The previous mock-based tests demonstrated internal consistency, not equivalence to the benchmark applications.

## 4. The unchanged verifier still confuses partial progress with whole-task success

At N=4, selected predictions give 62 true positives, 15 false positives, 30 false negatives, and 193 true negatives. Precision is 80.5% and recall is 67.4%. These improve substantially over the original run, but 28 real successes still have exactly zero confidence.

196/300 tasks have all-zero candidate confidence. This is not explained solely by error candidates: 154 of those tasks have at least one non-error candidate. Of 873 non-error candidates, 460 have no per-check breakdown, consistent with early deterministic rejection of predicted state; exact rejection reasons were not saved.

11/15 selected false positives have whole-task score zero yet confidence at least 0.5. The active formula is `0.75 * atomic_score + 0.25 * whole_score`; it measures partial check satisfaction, not the probability of passing every required check. As before, nominal verifier_samples=5 does not mean five whole-task votes in the active per-check branch, which explicitly uses samples=1.

Priority: keep a partial-progress ranking score separate from an all-check success prediction, and treat unsupported/unknown simulated state as unknown rather than a proof of native failure.

## 5. The remaining representation/budget boundaries still matter

- Full file evidence now reaches the Python helper, but final output remains capped at 20,000 characters. Ten final candidate errors explicitly hit that limit. Removing upstream truncation exposed this downstream bottleneck; the previous repair was incomplete across the full data path.
- The clean-read inventory optimization assumes a directory. Valid commands involving specific files such as `ls /testbed/data/salary.xlsx` can send a file path to directory-only `list_state`, yielding `[directory does not exist]` and aborting the rollout. Three final errors identify file operands (`salary.xlsx`, `invitation_letter.pdf`, `A12345homework.docx`).
- Two grounding failures attempt READ on `/testbed/calendar`, a directory. Malformed tool queries need recoverable, typed errors; immediate whole-rollout failure prevents the model from correcting READ to LIST.
- Python READ still prefers `read_raw`, and predict-all LIST normally returns pristine inventory; generic post-write reads still depend on model-applied transition history. Restoring symbolic Word/Excel shortcuts did not create a coherent overlay for every resource/app.
- Commit still replays a fixed sequence without adapting to real observations, so wrong simulated values continue to affect downstream actions.

## What the evidence supports

The prior repairs made a large measurable improvement, especially in file discovery. The dominant remaining bottleneck is now the read-program/data/tool contract, followed by app-semantic mismatches and verifier scoring. Increasing N or write-token budgets alone will not address these failures.

Recommended next sequence: typed read results and supported helpers; native-vs-symbolic differential checks; matching end-to-end evidence/output budgets and file-aware listing; then calibrated success verification. Before another full sweep, check representative XLSX, DOCX, PDF, calendar and file-conversion trajectories with both predicted and actual observations retained.

Limits: only selected candidates have official outcomes. The archive lacks full per-step predicted/real observation pairs, generated Python, and intermediate failed-attempt details. Consequently this audit cannot compute oracle pass@4, exact action-level fidelity, or the fraction of all 208 N=4 task failures caused exclusively by simulation rather than solver/selection errors. No production code was changed during this audit.
