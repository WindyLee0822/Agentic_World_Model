# R22: relaxed R16 with request-scoped simulator failure boundary

Status: 13 focused tests, 26 regression tests, and native execution check passed; full585 BoN16 job submitted and starting: `dw-appworld-ours-relaxed-r22-n16-full-20260913` (code `a90114c7-8565-4f3a-8d2e-0d136b32fe21/versions/1`).

This inherits the R20 relaxed setting: the simulator generates setup/read/write/error observations, decides write acceptance, and scores candidates. Deterministic completion deferral, answer-key rejection, optimistic first-write success, and hand-built parameter-error shortcuts are disabled. API docs, schemas, arguments, frozen time, read-only grounding, and ordered logical history remain available to the simulator. Native replay is limited to the selected candidate.

R21 introduced an error boundary to keep internal simulator failures out of solver observations, even when solver code catches exceptions. Its wrapper also changed errors during verification, bypassing the existing conservative zero-score recovery and discarding candidates. R22 applies prediction-error conversion only while an observation requester call is active. Verifier failures retain their original behavior. This does not relax the all-predicted API observation requirement.

Only src/tts/env/appworld_ours_failure_boundary.py differs from R21. Its installation remains inside the existing AppWorld/Ours/predict-all/model-driven opt-in branch. Other Python source is byte-identical; this is source isolation, not an empirical cross-dataset performance claim.

Validation: 13 focused tests and 26 staged prediction/routing regressions passed. Native execution check passed: solver-caught plain model error is withheld at the host boundary. R21's four native cases are prior evidence, not substituted for R22 results.

No approximately-63% BoN16 result is established yet. R16/R18 historical full runs were N2. R20/R21 records must remain separate from R22; cancellation requested for their confirmed source defects. See superseded-jobs.json and ../r21/failures/21abae1_2/analysis.json.
