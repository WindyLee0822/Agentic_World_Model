# Full repaired GPT-5.4 OfficeBench reliability audit

Job `dw-officebench-gpt54-repaired-full-n124-20260922` completed. All 900 requested records are present: 300 subtasks at each of N=1,2,4. Analysis uses the downloaded records, commit traces, error logs, frozen submitted source, and three isolated code reproductions. No candidate actions were executed for this audit; production code was not changed.

The main remaining bottleneck is the connection between predicted state and native success checks. Several deterministic implementation defects cause valid trajectories to receive zero confidence. There are also substantial observation-contract gaps beyond the four-task smoke subset.

## Results

| N | Native successes | True positives | False positives | False negatives | True negatives | All-zero pools |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 133/300 (44.3%) | 53 | 1 | 80 | 166 | 245/300 |
| 2 | 134/300 (44.7%) | 56 | 1 | 78 | 165 | 240/300 |
| 4 | 140/300 (46.7%) | 63 | 1 | 77 | 159 | 232/300 |

At N=4, success-prediction precision is 63/64 = 98.4%, but recall is only 63/140 = 45.0%. Selecting among more candidates has limited benefit when 77.3% of candidate pools have only zero scores. N=1 to N=4 gains eleven tasks and loses four, for seven net gains (2.3 percentage points).

There are **4,684 exact matches among 5,778 selected-action observation comparisons (81.1%)**. These are repeated selected-action executions across N, not 5,778 independent tasks or labels for unselected candidates. Exact disagreement includes wording, diagnostic output, and harness-added feedback; it is not equivalent to an incorrect state transition in every case.

There were zero reported simulator writes and 895 real commits. Five task/N cells had no viable candidate to commit. Runtime preflight passed Word/Excel PDF generation, extracted content, shell listing, and network denial.

The N=4 candidate records report 12,727 Python calls and 927 Python failures (7.3%), with 11 error candidates out of 1,200 candidate slots. Failed Python attempts can be repaired and are not the same as failed trajectories. Do not sum N-specific counters as independent calls: candidate pools are reused across N. The raw error log contains 916 entries; contract-validation failures and other accounting paths mean it is not identical to the record counter.

## Priority 1: repair evidence routing and format typing

**Email evidence is wired to the calendar.** `state_files_from_config` maps any check with `username` and no `file` to `./calendar/<username>.ics`, including `doc_type=email`. `_predicted_state` iterates only those requested specs. The verifier is instructed to look for `emails/<username>`, but that resource is not explicitly included in its final-state input. This defeats the mailbox reconstruction repair.

Concrete example: **3-12/0**, which was used in the smoke work, has no selected-action observation mismatch in the full run and succeeds natively. At N=4 it receives confidence 0 with **1 MET and 4 UNKNOWN** checks. The earlier separate scorer probe explicitly constructed mailbox entries, so it did not test this broken end-to-end handoff. Forty-nine dataset tasks use email checks; seventeen N=4 false negatives belong to that group. These are overlapping associations, not seventeen individually proven causal attributions.

**Existence checks accidentally demand plain text.** A `evaluate_file_exist` check on `data/result.docx` without `doc_type` becomes a `txt` spec. `_predicted_state` then marks a Word-produced DOCX as incompatible with plain text/PDF. A valid native document can therefore receive a deterministic failure before the per-check judge runs. Sixty tasks have an untyped DOCX existence check, including seventeen N=4 false negatives. This also affects selection priority via `hard_invalid`.

Both defects are reproduced from the frozen submitted functions in `contract_reproductions.json`, using synthetic copied inputs and no native actions.

**Repair:** distinguish existence-only requirements from decoding requirements; route email, calendar, ordinary files, and paired-file checks explicitly. Infer artifact format from the actual modeled producer/extension where appropriate. Assemble the final-state context once and validate that every check has its required operands before invoking the verifier. Test the complete predictor → evidence assembly → verifier path.

## Priority 2: make native-check computation typed and faithful

Of 916 logged Python errors, **637 arise during success verification**. Two repeated errors account for 627 entries:

- **352:** spreadsheet match records are dictionaries, but generated code performs `if m not in content`, treating each dictionary as a substring. The actual schema is `{row, col, value}`; native matching constructs the literal `(row, col): value`.
- **275:** `evaluate_diff_contain_text` is generated as a one-file/gold equality comparison. Its native contract instead uses `input_file`, `output_file`, and keywords in a unified diff. Looking for `file`/`result_file` yields None and then a string-concatenation error.

These are schema/semantics errors in generated programs, not corruption introduced by Python string operations. All fourteen natively successful N=4 tasks that use `evaluate_excel_cell_value` are false negatives.

**Negative checks are changed by atomic splitting.** `_atomic_requirements` splits every short keyword list. Native `evaluate_not_contain([a,b])` means `NOT(contains(a) AND contains(b))`; requiring both split negative checks means `NOT contains(a) AND NOT contains(b)`. They differ. An isolated counterexample passes the native predicate and fails the split predicate. Seven tasks have this configuration, including successful false-negative cases 1-20/0 and 2-40/1.

N=4 false-negative breakdown: **41 have UNMET checks, 26 have UNKNOWN checks, 9 have no per-check audit, and only 1 has all checks MET with a low whole-task score.** The current dominant problem is upstream of whole-task voting.

**Repair:** use tested pure evaluator adapters over predicted/copy-only state, or constrained generated programs with explicit typed schemas and validated operators. Preserve native predicate composition and diff semantics. This does not execute a candidate against the real environment: the observations and final state remain predictions. Additional votes cannot repair a systematically wrong predicate or absent resource.

## Priority 3: extend action contracts to the uncovered apps and shell operations

| Action | Exact / compared | Main observed gap |
|---|---:|---|
| shell.command | 946/1,443 | Wrong current existence, relative paths, post-write listings, wrappers, grounding refusals |
| email.read_email | 0/146 | Wrong/missing single-message evidence and native MIME/header/body rendering |
| llm.complete_text | 0/112 | Simulator predicts useful text while native app fails |
| word.create_new_file | 234/321 | Unconstrained response wording plus false missing-parent/current-state claims |
| excel.create_new_file | 106/152 | Response templates and path resolution |
| pdf.convert_to_image | 0/33 | Native response is `Success`; simulator invents a longer template |
| pdf.convert_to_word | 0/27 | Missing-input hallucinations and native conversion diagnostics/timing |
| word.convert_to_pdf | 80/80 | Prior repair holds for observed calls |
| excel.convert_to_pdf | 12/12 | Prior repair holds for observed calls |
| email.list_emails | 134/134 | Prior mailbox-list repair holds |
| calendar.create_event | 258/258 | Correct action response does not ensure correct reconstructed ICS state |
| excel.set_cell | 1,544/1,549 | Largely faithful; a few wrong precondition predictions remain |

Single-email reads have no equivalent operand-grounding branch to Excel/Word/PDF/OCR/calendar, and `read_target` identifies the mailbox directory rather than the selected `email_id`. The list-emails repair therefore did not repair read-email.

For shell actions, **1-3/0** illustrates semantic damage: `mv` is predicted to fail because its source is supposedly absent, while the native move succeeds. Subsequent simulated state and the success score remain wrong. Directory creation and relative paths cause similar cascades. A further 76 mismatches contain a grounding-policy refusal as the predicted environment observation, including 74 shell calls. The refusal says which inspection tool was unavailable; it is not a forecast of the native shell response.

For the native LLM app, **88/112 calls fail because `/openai_key.txt` is absent**; the other 24 fail in shell/CLI parsing. Its bundled implementation uses a separate legacy OpenAI client/model, independent of the GPT-5.4 actor endpoint. Supplying the actor endpoint alone does not provision this app. Expose service-availability and CLI-contract metadata to the simulator without revealing credentials. If the runtime app itself is enabled or replaced, identify that as a benchmark-environment change and validate it separately.

**Repair:** add pure single-email rendering and logical path/filesystem transitions; materialize input existence as well as output targets for all mutation families. Supply public native templates and exact CLI argument semantics, including sender names containing spaces. Keep uncertain conversion content distinct from known output existence. Separate nondeterministic conversion diagnostics from stable semantic observations using a documented observation contract.

## Priority 4: prevent uncertain prechecks from discarding good incumbents

Four tasks pass at N=1 and fail at N=4: **3-8/3, 3-22/0, 3-65/0, 3-84/0**. All candidates in each task have zero confidence and `done=True`.

The code ranks `not hard_invalid` before the zero-confidence branch. The incumbent may therefore be excluded from `contenders` before the guard runs; the guard starts from `native_best`, not necessarily the original incumbent. In 3-65/0 it replaces the successful candidate with a trajectory that only lists files and finishes, despite the new audit having 0 MET, 1 UNMET, and 2 UNKNOWN checks. The recorded guard message claims replacement requires a strictly better complete audit, which does not describe this earlier priority replacement.

**Repair:** distinguish malformed/unexecutable trajectories from a predicted task failure or uncertain reconstructed state. Keep the incumbent in a zero-score comparison unless a sound structural invalidity proof applies. Require positive, comparable, complete evidence for a progress-based replacement. Log the precise precheck and priority reason, not only the final guard message. Fix the evidence bugs first so selection is not optimized around corrupted scores.

## Priority 5: separate harness feedback from native observations and improve diagnostics

- **57** mismatches contain the harness's synthetic repeated-action nudge, including a 200-character truncation. That feedback is stored as an app observation, but real replay executes the app action normally.
- **69** malformed-action records predict the harness's format error, whereas replay submits a `{raw: ...}` record and receives `KeyError: 'app'`.
- **67** mismatches retain a `PREDICT:` prefix inside the observation payload. Counts overlap with other categories.
- The actor's default action-output budget is 1,024 tokens. Long Word-write actions occur among malformed cases, but the logged raw action is itself truncated, so the records cannot establish whether the model hit that limit. Preserve complete actions and log the model's finish reason to distinguish truncation from other parse errors.

**Repair:** represent policy feedback, parse errors, grounding errors, and app observations as different event types. Preserve exact app observations for comparison. Store per-candidate predicted final-state resource schemas, per-check verdicts/reasons, precheck failures, and transition provenance; commit traces also need N, candidate index, and step. Current records omit enough of this information that every false negative cannot be causally assigned from logs alone. Keep copied private values out of generator prompts; comparative reranking also deserves a privacy audit because its current prompt interpolates predicted final-state text.

## Benchmark defects and measurement limits

The sole false-positive task, repeated at all N, is **3-82/0**. It asks for `list.docx` but evaluates containment with `doc_type=xlsx`. The predicted text has the requested names; the native evaluator attempts the Excel reader on a DOCX. An isolated reader probe confirms that openpyxl rejects the `.docx` extension with `InvalidFileException` before reading the file (`docx_excel_decoder_probe.json`). Treat this as both an evaluator/configuration inconsistency and a missing decoder-compatibility check in the forecast. Keep the original native result; report any corrected benchmark separately.

The previous four-task result of 117/117 exact observations was real but did not cover these failure families. Its separate successful scorer probe supplied mailbox state directly and therefore bypassed the email evidence-routing defect. Full-run success recall and action coverage provide a much stricter validation.

Unselected candidates have no native outcome labels, by design. This audit does not compute a native oracle over all N candidates or claim that every zero-score alternative would succeed. The four regressions are established using selected outcomes across N; broader ranking quality remains partially unobserved.

## Next validation set

Prioritize evidence wiring/type fixes and native-check adapters first, then shell/single-email contracts and the selection guard. Reuse known regressions and false negatives: 3-12/0 (mailbox handoff), 1-8/0 (gold-backed Excel state), 1-4/0 (cell-value checks), 1-20/0 (negative predicate), 1-3/0 (move state), 3-65/0 (zero-score replacement), plus one diff-contain task and 3-82/0 as a decoder-mismatch control. Include each evaluator function and every action family in coverage, rather than selecting only conversion/email successes.

Success criteria should separately measure exact observations, semantic/state agreement, per-check accuracy, success recall/precision, and selected-task success. Preserve predict-all → select-best → commit-one. Raise N only after zero-score saturation and these systematic errors are reduced.

## Evidence

- `audit.json`: full metrics, overlapping task associations, and regression records.
- `records.jsonl`, `run.log`, `runtime_preflight.json`: downloaded job artifacts.
- `observation_mismatches.json`, `python_errors.json`: extracted mismatches/errors.
- `false_negatives_n4.json`, `false_positives.json`: complete affected records.
- `contract_reproductions.json`: isolated frozen-code reproductions.
- `../audit_full.py`, `../diagnose_contracts.py`: reproducible audit scripts.
