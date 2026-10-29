# Completion audit: incomplete

- Relaxed setting implemented; same inputs and read-only grounding available to simulator.
- Observation boundary: 13 focused tests passed, including nested failure suppression, predicted rejection recovery, and actual verifier zero-score handling. Native check passed. R20 response-routing review applies with R22 boundary correction; full runtime review remains required.
- Selection unchanged: simulator scores, native replay only selected candidate. Full-run record audit pending.
- Isolation: only opt-in AppWorld/Ours boundary changed versus R21; all other staged Python files identical (isolation.json).
- Benchmark target: approximately 63% on full585 BoN16. Fresh run submitted: dw-appworld-ours-relaxed-r22-n16-full-20260913, running; driver confirms full585 benchmark command with N16 and 16 workers. Full585 evaluation pending; no measured final claim.
- Failed-candidate placeholder setup metadata requires review, not automatic inference of observation bypass. Keep audit flags visible.
