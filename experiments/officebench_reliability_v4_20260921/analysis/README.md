# Final focused validation

Job: `dw-officebench-gpt54-reliability-v4-focused-n124-20260921`.

This follows the 12-task v2 smoke run. The intermediate v3 submission was canceled during startup and superseded by this job. The final job runs task index 20 in all three levels: `1-5/1`, `2-15/0`, `3-12/0`, each at N=1,2,4. These are deliberately selected diagnostics, not an unbiased benchmark estimate.

In addition to the v2 repairs, this source:

- Preloads typed immutable operands for clean Word, Excel, PDF, OCR and calendar reads. The world model computes every observation; no candidate app action is executed.
- Applies native driver-style observation extraction and empty shell-output wording, and preserves complete observations for committed-trajectory comparison.
- Rejects query envelopes masquerading as final text observations, invoking bounded local program repair instead.

The relevant suites pass: **156 passed, 6 skipped**. Tests include native function comparisons on disposable files, configured verifier voting, large outputs, file/directory queries and malformed-final-result repair. See `../tests.txt`.

The same immutable-evidence → simulated candidate trajectories → selection → one real commit per task/N paradigm remains in place. No unselected candidate is evaluated in the real environment.

This run was canceled during repeated real replays. The retained partial_run.log supports trace-level diagnosis; no aggregate task-success result is claimed. See ../../officebench_reliability_final_20260921/report.md for the completed pilot and final bounded check. Early v2 traces additionally expose semantic prediction mismatches and a real LibreOffice launch failure (`/usr/bin/oosplash: not found`) during PDF conversion. These must not be conflated with Python protocol failures or claimed resolved merely because candidates finish.
