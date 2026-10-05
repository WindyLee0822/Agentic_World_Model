# R21: simulator failure boundary (submitted)

R20 can send simulator RuntimeError text back to the solver through AppWorld world.execute. R21 stages an opt-in outer execution boundary. Non-plain requester exceptions poison the candidate; the host raises before returning cell output, even if solver code catches the inner exception. Intentional predicted API rejections retain AppWorld plain Exception behavior. Existing predictor evidence access, scoring, and relaxed goal checks remain as in R20.

Apply this overlay after R20. The only integration change is under the existing Ours/predict-all/model-driven opt-in branch. The live R20 source asset is unchanged.

Validation: all 10 tests in test_failure_boundary.py passed. The suite exercises exception rendering, solver catch/retry, predicted rejections, and normal solver Python errors using an execution-boundary test double. check_native_boundary.py also passed three cases with native AppWorld: uncaught internal error, solver-caught internal error, and preserved predicted rejection (native-boundary-checks.json). The plain model-exception distinction also passed native integration (native-plain-error-checks.json). Staged prediction/routing regression suite: 26 passed. Job dw-appworld-ours-relaxed-r21-n16-full-20260913 was submitted with code asset fa24e82c-45e5-41b8-bbf3-ee3758e07c60:1. Isolation is recorded in isolation.json. Pending: remaining observation-route audit, full585 N16 accuracy and runtime fidelity audit. No final performance claim.


Superseded and canceled on 2026-09-13: the R21 boundary also wrapped verifier errors, bypassing existing zero-score recovery. R22 fixes request scoping. Last cancellation-time audit: 10 completed, 8 successful, 1 placeholder metadata flag; incomplete, not a full benchmark result. See ../r22/README.md.
