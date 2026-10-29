# R20 API response routing review

Reviewed the staged R20 source in `/tmp/appworld-r19-resume-stage` (the directory name predates its R20 overlay). This is a source review, not full-run runtime proof.

| Route | Simulator boundary |
| --- | --- |
| Documented ordinary reads | `install_wm_requester` calls `predict_read` when predict-all is enabled. Both clean and dirty reads invoke the generated-program path. |
| Ordinary writes and download endpoints | Requester calls Ours `predicted_write`, which dispatches to the experimental `predict_write`. The simulator supplies the outcome, status and body; malformed envelopes raise an operational error. |
| Supervisor, API documentation and login | These are precisely the base requester's passthrough sets. The outer setup wrapper suppresses native printing and uses the inner response as evidence for `predict_read`, returning the generated result. Logical completion responses also pass through this wrapper. |
| Absent API metadata | The experimental predictor's `predicts_request_errors` activates the setup wrapper for this route. The inner blocked-request body or plain API exception becomes prediction evidence. |
| Predicted HTTP failures | Requester formats or returns the simulator's predicted error, depending on `raise_on_failure`. The setup wrapper separately predicts plain native setup API errors. Infrastructure exceptions propagate as failures. |
| Read cache | Ours `predict_read` calls its superclass before caching the generated result. Cache lookup occurs in the internal `_read_write_precondition` evidence function, keyed by applied-write count and hydrated request arguments. It does not bypass agent-facing prediction. |
| Exhausted prediction budget | Ours `_over_budget` raises `WMHardTimeout`; the base class's fallback-return branch is unreachable through that override. |
| Unusable read JSON | Ours `_predict` rejects parsed `None` in observation modes before returning to base `predict_read`. Thus the base parser's fallback is not reached for that case. This also rejects literal JSON null rather than predicting it; setup evidence is JSON-encoded to avoid that ambiguity. |
| Hand-built dirty reads | Program-only mode disables the base Venmo and Amazon dirty-read shortcuts. Ours also disables the inherited grounded Venmo funding shortcut. The setup wrapper rejects the deterministic grounded-clean-read option. |

No new observation bypass was identified in these reviewed routes. The review does not establish every API's semantic fidelity, adequate evidence under all budgets, or correctness of every generated program. Full benchmark and runtime audits remain pending as recorded in `completion-audit.md`.


## Additional finding: simulator operational errors (2026-09-13)

R20 is not yet compliant with the literal all-observation requirement. The opt-in predictor raises RuntimeError on unusable model programs or invalid response envelopes. The ReAct loop calls world.execute(code) and then unconditionally appends its returned output as a user observation (appworld_react_agent.py:606 onwards). AppWorld can render these exceptions as execution output. Thus propagating an internal exception does not by itself prevent agent exposure. Task 325d6ec_1 has solver comments describing repeated errors/retries and 40 selected-candidate Python failures; exact raw earlier error text has not been recovered. This is a source-level route gap regardless of that task's precise error text. Zero checkpoint flags do not establish absence of this leak.

Required follow-up: isolate simulator operational-failure handling to the AppWorld/Ours opt-in path, retain predicted API rejections as normal observations, and prevent internal prediction failures from reaching the next solver turn. Add meaningful integration coverage for the world.execute exception-to-output boundary, including solver try/except. Preserve the live R20 run as measurement evidence; do not claim it proves final fidelity.
