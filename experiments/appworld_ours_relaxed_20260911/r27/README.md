# R27: Verifier state instructions and diagnostic coverage

R26's successful payment task still received score zero. Its generated final verifier searched pristine read pages for a newly created record before applying simulated transitions. The truncated program prevents proving the exact executed failing predicate, but exposes a concrete final-state reasoning risk.

R27 explicitly tells the opt-in AppWorld/Ours verifier to apply ordered successful writes to pristine grounding evidence before checking final state. It does not fill in state, override scores, or count unknown checks as success. The model still generates observations and verdicts; only the selected candidate is replayed natively.

Host-only diagnostics now retain up to 32,000 response characters, program execution errors, and propagated prediction errors. JWT-like tokens are redacted; input context values are not dumped. Diagnostic write failures do not replace model output. No diagnostic event is appended to solver observations.

The same two N=1 normal/L1 tasks (indices9,34) test file prediction failures and payment verification. This is diagnostic only; the full BoN@16 target remains unproven. R22's full run continues unchanged.

## Completed diagnostic (2026-09-14)

Azure job `dw-appworld-ours-relaxed-r27-n1-probe-20260914` completed with both targeted tasks successful in native evaluation. Both candidates received simulator score 1.0, recorded setup prediction enabled, made one selected-candidate native commit, and recorded zero simulator writes. See `completed-analysis.json` and `probe-records.jsonl`.

The payment task (`024c982_2`) recorded 38 selected Python calls and 3 Python failures; the file task (`5a83b05_1`) recorded 325 calls and 27 failures. Failed earlier attempts are visible in the diagnostic traces despite the final candidate error being null. The file task's successful final verifier compares simulated deleted paths with expected paths; context-key and sandbox errors still consume repair calls.

These two N=1 successes do not establish BoN@16 accuracy or reliable ranking. In particular, `verifier-counterfactual-audit.json` shows that the payment verifier accepts a synthetic wrong-recipient case: it derives the receiver ID from gold instead of validating the candidate recipient. This is a scoring limitation, not evidence of native replay of an unselected candidate. Full-benchmark and routing/isolation completion remain pending.
