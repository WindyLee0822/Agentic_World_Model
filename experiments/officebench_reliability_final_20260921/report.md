# OfficeBench reliability repairs: implementation and validation

Implemented the requested read-interface, local-repair, verifier-score and representation/budget repairs. Candidate actions remain simulated from immutable evidence; only each selected trajectory is replayed in the real environment.

## Completed 12-task diagnostic

Run: `dw-officebench-gpt54-reliability-v2-smoke-n124-20260921`. All 12 tasks and all 36 task/N records completed before the stop request took effect. Azure reports Completed. This intermediate version predates the later observation-format, strict-final-schema, operand-preloading and grounded-creation refinements.

Comparison uses the same 12 tasks from `dw-officebench-ours-llm-wm-gpt54-repaired-n124-20260918`. This is a deliberately targeted sample, not a full-benchmark estimate.

| N | Previous successes | Repaired successes | Previous exhausted candidates | Repaired exhausted candidates | Previous selected false positives | Repaired selected false positives |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2/12 | 3/12 | 4/12 | 0/12 | 0 | 0 |
| 2 | 2/12 | 3/12 | 5/24 | 0/24 | 1 | 0 |
| 4 | 2/12 | 3/12 | 11/48 | 0/48 | 1 | 0 |

At N=4, 11/12 tasks still had all-zero candidate confidence. Two of the three real successes were predicted negative. Thus eliminating exhausted program errors and partial-score false positives does **not** establish good end-to-end reliability or calibrated confidence. The intermediate pilot recorded zero simulator writes and exactly one real commit per task/N cell.

## Repairs in the final source

1. Explicit decoded read formats and Python types; inventory root existence/type; pristine-versus-predicted-state provenance; native driver observation normalization; safe regex match methods; immutable operand preloading. Query envelopes cannot masquerade as final observations.
2. Bounded local program repair gets the failed code, error and read schemas. Persistent program failures do not replay the same whole trajectory indefinitely. Exhausted failures retain counters; diagnostics capture generated code.
4. Whole-task confidence no longer blends partial-check coverage into success. Unmet requirements gate to zero; unknown requirements cap confidence below 0.5. Whole-task voting honors the configured sample count. Partial coverage is retained separately. The legacy whole_weight setting still appears in run logs but is no longer used to blend success confidence. Incorrect Word append prechecks were removed.
5. Input and output budgets both permit 2,000,000 characters and fail explicitly at the bound. READ of a directory is a recoverable query error; LIST supports file operands. Complete observations and selected-commit traces are retained.

Native semantic corrections additionally cover Excel Fire conversion (including 1/True clearing and None preserving a cell), Word append/quote stripping and DOCX-versus-text representation, native cell rendering, and creation dependencies. Word/Excel creation is now a pure state transition grounded in immutable directory inventory and prior predicted transitions. It never calls the native app for a candidate.

## Follow-up evidence and limitations

- The v4 focused run was intentionally canceled during repeated real replays. Its retained commit traces exposed calendar rendering differences, false missing-file/directory predictions, and incorrect post-write state. No full-task aggregate is claimed for this canceled run.
- The completed v5 eight-step probe showed that correcting the input footprint and prompt alone was insufficient: Excel creation was still predicted to fail because its parent was guessed missing. That failed check motivated the final grounded creation transition and the explicit LIST schema.
- The real runtime also failed PDF conversions with `/usr/bin/oosplash: not found`, while its app wrapper still printed success. Some `ls /testbed` calls reported a missing directory even though `find` located files beneath it. These are observed runtime inconsistencies; their underlying cause has not been proven or repaired in this change.
- Native app wording, calendar timestamp rendering, generic post-write state reasoning and synthetic blocked/repeated-action observations still require work. Preserve the predict-all paradigm when addressing these; executing unselected candidates is not a remedy used here.

## Validation

**163 passed, 6 skipped** across the relevant OfficeBench, native differential, agent, remote-job and selection suites. Native differential tests use unchanged benchmark function bodies on disposable files and explicitly assert that prediction does not create the candidate output. The final source manifest verifies all 53 source files.

Final bounded live check: `dw-officebench-gpt54-grounded-creation-20260921`, task `2-15/0`, N=1, eight productive steps. This checks transition fidelity, not full-task completion. Azure reports **Completed**. The final check recorded six native actions, zero Python failures, zero rollout errors, zero simulator writes and one real commit. Creation matched the native observation exactly. All five write actions correctly predicted success, while their wording matched exactly only for creation. The calendar-read observation still differed. Overall, one of six observations was an exact string match.

The verifier recorded two met and two unmet checks: partial coverage 0.5, whole-task confidence 0.0, and actual full-task success false. This deliberately short prefix does not complete the agenda and is not a full-task success-rate estimate. It verifies the corrected creation transition and the separation of partial coverage from success confidence.

**Conclusion:** the requested interface, repair, scoring and budget mechanisms are implemented and tested. The creation regression is fixed in the final live check. The simulator is still not reliable enough end to end: the completed pilot improved from 2/12 to 3/12 selected successes, and generic state reasoning, output fidelity and native runtime inconsistencies remain unresolved.

Artifacts: [Completed pilot metrics](/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/officebench_reliability_v2_20260921/analysis/metrics.json), [Final transition check](/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/officebench_reliability_final_20260921/analysis/transition_check.json), [Final selected observations](/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/officebench_reliability_final_20260921/analysis/selected_observations.json), [Final source changes](/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/officebench_reliability_final_20260921/analysis/changes.patch).

[Final Azure job](https://ml.azure.com/runs/dw-officebench-gpt54-grounded-creation-20260921?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/external-collaborations/workspaces/M365Research-External-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47).
