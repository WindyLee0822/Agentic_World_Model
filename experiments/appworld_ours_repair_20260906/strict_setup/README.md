# Strict setup-observation experiment

R16 completed at 367/585 official BoN@2 successes (62.74%), but retained exact
setup observations. This experiment also routes setup responses through
simulator-generated Python. The R16 runtime under the repository src/ is preserved.

The Ours-only wrapper runs after install_wm_requester. Native setup responses
provide evidence; only generated results reach the solver. Synthetic credentials
are preserved through JSON-encoded evidence without changing shared redaction.
Native keyword binding is supported. Logical completion remains simulator-local.
API errors are predicted; infrastructure failures propagate without fallback.

Validation completed:

- 12 routing and actual-program tests passed (one AppWorld deprecation warning).
- Three real DeepSeek-V4-Pro synthetic setup cases passed.
- Target-version native task 024c982_1: profile, credentials and API docs matched;
  generated credentials supported login inside world.execute; logical completion
  did not change native supervisor state. All cases used generated Python.

The integration is staged in /tmp/appworld-strict-stage, with durable changed-file
copies under staged_changes/. It adds candidate setup counts and an optional
strict setup requirement to the full-results gate. The native N2 rollout passed officially: setup counts 17 and 21, simulator
selection, one official commit and zero simulator writes. The focused suite
passed 53 tests. Nine shared B1 callback failures remain unchanged.

Full-585 evaluation submitted as dw-appworld-ours-strict-r17-full-20260907
(code asset 2abdb0fb-9282-46e4-ab3a-e33f4c1daeba:1); initial status Starting. R16's score is not a measured
score for this new implementation.

Environment: isolated /tmp/appworld-strict-env, pinned AppWorld 0.2.0.dev0 commit
a072b7a86e7c1d5b1d7175659d750ebb9b79f10a; matching data 0.2.0 under
/tmp/appworld-strict-data. The older base installation was not modified.
