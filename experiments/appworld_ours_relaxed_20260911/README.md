# R19: relaxed AppWorld Ours, BoN@16

Submitted job: `dw-appworld-ours-relaxed-r19-n16-full-20260911`.
Target: approximately 63% official task success over all 585 test tasks. R19 failed after 44 normal/L1 tasks (40 successes); this is not an overall success estimate. See `failed-run-tail.log` and `r20/README.md` for diagnosis and the replacement evaluation.

This is an isolated experimental setting, reconstructed from R16 plus the R18 setup-observation fixes. The main runtime and R16 are unchanged. The new predictor is selected only for AppWorld Ours when `TTS_APPWORLD_OURS_MODEL_DRIVEN_WRITES=1`.

Changes:
- Every ordinary write uses the simulator to choose success or rejection and generate the response. Removed the optimistic first-time/simple-response route and the hand-built missing-parameter error route in this opt-in predictor.
- Simulator context includes full call arguments, API documentation, success/failure schemas, frozen time, call history, successful logical writes, and inherited read-only evidence tools. Write precondition reads account for prior simulated writes.
- Invalid simulator output fails the candidate rather than synthesizing an API observation.
- `TTS_APPWORLD_COMPLETION_GUARD=0` removes the manually imposed completion deferral for this job. R18's wrapper still generates the completion observation through the simulator.
- R18 setup observation prediction, including suppression of native `show=True` output, is retained.
- Candidate ranking remains simulator-based. Native candidate writes are prohibited; only the selected trajectory is replayed for official scoring.

Retained constraints/evidence: execution budgets, read-only grounding restrictions, and the R16 verifier's reference checklist/gold inputs remain. This is not a reference-free selection experiment or a claim that all runtime exceptions are model-generated. Operational failures abort candidates. Setup prediction metadata for aborted candidates can require review.

Validation: 34 focused checks passed in `/anaconda/envs/appwm`, including actual generated Python execution for success and rejection, missing arguments reaching the simulator, invalid output not producing a fallback success, logical write recording, setup routing, and result audits. Shared base predictor and selection source hashes match the main checkout; other datasets have not been rerun. The full benchmark runs in the pinned AML AppWorld environment.

`submission.json` identifies the immutable uploaded code asset. `staged_changes/` holds the R19 changes and R18 setup wrapper. `stage.json` records source hashes and reconstruction inputs. Run `poll.py` with `/opt/az/bin/python3` to refresh AML status and available official records. Partial rates are provisional; approximately 63% is a target, not a verified outcome.
