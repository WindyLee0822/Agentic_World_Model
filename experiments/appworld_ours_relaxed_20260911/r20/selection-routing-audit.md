# R20 candidate selection review

Reviewed staged source `/tmp/appworld-r19-resume-stage` and the 45 completed checkpoints available at 2026-09-12 21:36 UTC.

- `real_appworld.py` assigns candidate confidence from `predictor.predict_success`, except operational hard timeout rejection. R20 bypasses the old deterministic answer-key and missing-completion rejection. The simulator retains reference-assisted verification.
- `harness.py` in its world-model-agent branch generates the candidate pool, then calls `Selector`, then commits the selected trajectory through `CommitChokepoint`. An invalid pool can produce zero commits. Official native evaluation happens after selection.
- AppWorld has no `refine_rollout_scores` implementation; the similarly named OfficeBench method is not part of this path.
- Current records use `simulator_success`. This scores each candidate by its predicted success flag. Selector also prioritizes operationally valid completed trajectories and breaks equal scores by lower cost then earlier index. It is therefore simulator-scored selection with deterministic validity and tie handling, not another LLM selecting an index.
- All 45 observed records have 16 completed candidates, no native candidate success labels, and a selected score equal to the maximum recorded score. The broader checkpoint audit checks simulator writes and commit counts.

This establishes source ordering and consistency of the current partial records. Records do not retain all ranking metadata, so the maximum-score check is not a complete independent reconstruction of every tie decision. Full-run runtime verification remains pending. Do not execute unselected candidates to obtain selection labels.
