# Final simulator refinement (2026-09-22)

Builds on the runtime/state repair in `../officebench_runtime_state_repair_20260922/`.

- Freeze each conversion input at conversion time, including prior edits. Supply it as copied Python context for final-state reconstruction and dirty reads. Later source edits cannot retroactively change the converted output.
- Explain the nested write-record schema; prohibit native conversion/file I/O in generated prediction programs. Reject pristine-output absence as proof of absence after a predicted conversion, while allowing explicitly unknown output.
- Parse email recipients with the same Fire literal semantics: `Bob, Tom` is a tuple and fails; a list action argument uses the native Multiple recipients rejection. Failed sends do not create a populated logical mailbox.
- Preserve native DFS/readdir ordering in immutable inventories so predicted find output matches the native environment.
- Detect a DOCX overwrite of an earlier PDF, even when the write history includes a prior conversion.

Validation: 134 affected tests passed, 6 skipped; 42 unaffected integration/selection tests passed previously. Live reconstruction probe passed 5/5 (one locally repaired argument-schema error); separate comma-recipient GPT-5.4 probe passed with zero Python failures and no applied writes. No native candidate action was executed during these probes.

Focused job: `dw-officebench-gpt54-final-state-n124-20260922`, levels 2/3, task indices 0/20, N=1/2/4. Native runtime preflight remains mandatory. Native evaluation is unchanged.

## Completed run and scorer follow-up

The job completed: 117/117 selected-action observations matched exactly; zero Python failures,
zero simulator writes, and 12 selected real commits. Native success was 3/4 at each N, up from
2/4 on the same task subset. See `report.md` for original scores and calibration limitations.

A subsequent scorer change addresses the 3-12/0 false negative: recovered failures do not veto
final success, and whole-task votes forecast the conjunction of native checks without adding
unlisted filename requirements. Both atomic and whole checks now inspect copied values in
Python; prompts contain the public schema/operation names, not file contents or gold values.
Legacy text/disabled-Python ablations remain available. See `verifier_followup.patch` and
`verifier_followup_source_manifest.json`; the original job's frozen source and scores are unchanged.

The regression suite after this scorer change passed 177 tests with 6 skipped
(`tests_verifier_followup.txt`). `verifier_followup.py` separately re-scores the frozen successful
3-12/0 trajectory, plus wrong-content and missing-evidence controls, without executing native
apps. Its reconstructed state is a logical projection of recorded predicted writes, not a fresh
end-to-end trial or a PDF-byte reconstruction. Earlier probe failures are retained in
`verifier_followup_attempt1.json` and `.log`.

Final follow-up audit passed all three cases: confidences 0.8/0.0/0.0, 25 Python calls, zero Python failures, and private-content prompt checks passed. See the final section of `report.md`.
