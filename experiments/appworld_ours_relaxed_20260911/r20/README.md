# R20: simulator-controlled errors and verification

Target remains approximately 63% official success on all 585 AppWorld test tasks at BoN@16. R20 has no measured accuracy yet. Do not merge R19 records into R20: verifier routing changed.

R19 ended Failed after 44 normal/L1 tasks, 40 successes, zero flags under its record audit. The run log reports the hang guard stopped the run after 7,200 seconds without a completed task/N cell. Completed tasks ranged from 472 to 19,979 seconds; the median was 1,189 seconds. This proves that the old watchdog was shorter than some legitimate N16 task runtimes, but does not establish whether every unfinished worker was healthy. R19's partial rate is not an overall benchmark estimate.

Changes beyond R19, restricted to the opt-in AppWorld Ours predictor:
- Missing-API-metadata errors from the inner read-only requester become evidence for a simulator-generated observation. Native error text is not returned as a fallback or printed by `show=True`.
- Deterministic answer-key comparison and missing-completion rejection no longer bypass the simulator verifier. The simulator receives the same checklist/gold and read-only logical-state evidence and judges success. Operational timeouts still invalidate candidates.
- The job uses the existing `--max-tasks-per-child 1` option (forkserver, fresh worker per task) and a 129,600-second no-completed-cell watchdog. This accommodates up to 16 candidates × 3 attempts × 2,400 seconds plus overhead. Candidate budgets are unchanged. Worker recycling is a mitigation; it is not a proven diagnosis of the R19 stall.

R19 model-driven write responses, R18 setup prediction, read-only grounding, and selected-only native replay remain enabled. Ordinary read identity/caching and generated-program protocols remain as before. Framework/operational failures are not claimed to be simulator-predicted API observations. This is not reference-free verification.

Validation: 37 focused tests pass, including actual generated Python for success/rejection, missing-metadata error generation in both exception and returned-body modes, no native stdout leak, and the new operational-rejection policy. `tests.log` preserves output. `isolation.json` compares shared source files to the main checkout; source isolation is not a measured cross-dataset performance result.

`submission.json` identifies the immutable AML code asset. `poll.py` refreshes authoritative job status and records. `staged_changes` and `stage.json` preserve reconstruction inputs and source hashes. The run must complete with full coverage and its runtime evidence must be audited before claiming the objective is achieved.

Goal-turn classification: progress—authoritative R19 failure and partial results retrieved; prediction and verifier gaps fixed and tested; replacement evaluation submitted. Goal remains incomplete.

The local result audit now checks exact official task identities and subset counts, rejects duplicate/extra/missing cells and wrong N, and validates simulator selection, no candidate native writes, selected replay counts, and setup prediction metadata. Thirteen audit checks passed, including corrupt-record cases and the actual partial R19 records. The audit reports distance from 63% without inventing a hard threshold for “approximately,” and never treats record checks alone as completion proof. These monitoring changes do not modify the submitted AML code asset.
