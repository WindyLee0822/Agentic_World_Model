# R29 reliability analysis

R29 completed all 585 tasks at N=1: 265 successes (45.30%) and 320 failures. This is a full N=1 result, not BoN@16. Analysis uses the final downloaded records and all 1,045 simulator trace files (145.7 MB). Machine-readable counts and record hash are in reliability-analysis.json.

## Main finding

There are two distinct reliability problems. The clearest directly measured loss at N=1 is failure to produce simulator observations: 146 tasks (24.96% of all tasks; 45.63% of failures) never reach native replay. The most severe problem for future best-of-N selection is low verifier sensitivity: 199 of 265 successful tasks receive score zero (75.09%). A simulator-generated program is not by itself a reliable simulator or evaluator.

Of the 439 tasks that reach replay, 265 succeed (60.36%) and 174 fail. The records alone cannot divide those 174 failures into incorrect simulated observations, solver mistakes, and incomplete task execution; it would be incorrect to attribute all of them to the world model. At N=1 there is no alternative candidate to rank, so bad verifier scores do not directly cause the 45.3% native success rate.

## Verifier scores versus native outcome

| Simulator score | Native success | Native failure after replay |
| --- | ---: | ---: |
| 1 | 66 | 17 |
| 0 | 199 | 157 |

Positive-score precision is 66/83 = 79.52%, but recall is only 66/265 = 24.91%. Among score-zero candidates that replayed, 199/356 = 55.90% actually succeeded. These are descriptive N=1 statistics, not a measured N>1 selection gain.

32 of the 199 false-negative tasks have a logged verifier prediction failure. The other 167 have no such logged failure: execution-error repair alone cannot explain away the scoring problem. These cases can include wrong predicates, wrong predicted state, insufficient evidence, and output interpretation; the diagnostics do not retain all input values or final Python stdout to isolate every cause. A representative executable verifier, task b6d1104_2, assumes dictionary insertion order is chronological month order and parses a particular note format. The native task passes but receives score zero; the trace exposes fragile checks, though it does not prove which predicate failed.

## Observation execution and repair overhead

Final candidate errors on the 146 aborted tasks are 141 SimulatorObservationFailure wrappers, four prediction-budget timeouts, and one exhausted HTTP-429 retry sequence. The generic wrappers hide underlying causes.

Across all attempts, traces contain 4,316 Python errors affecting 511 tasks:

- 780 disallowed json.JSONDecodeError attribute accesses.
- 545 disallowed __name__ accesses.
- 454 undefined lowercase false and 326 undefined lowercase true names.
- 304 failures exceeding 100,000 executed lines.
- 180 missing json names and 143 oversized integer literals.

These counts include recoverable errors and repeated attempts; they must not be added as independent failed tasks. For example, 187 line-limit failures contain deepcopy(data), across 52 tasks. Copying the full accumulated context can exhaust the execution budget before substantive prediction. An oversized-integer example transcribes a payment-card number into Python source rather than accessing the copied read value. A lowercase-false example emits a JSON-style Boolean inside a Python read-query dictionary.

The final records sum to only 1,531 Python failures versus 4,316 in all traces: failed-attempt costs and errors are not fully reflected in candidate summary counters. This is a measurement blind spot, not evidence those errors did not occur. The 146 record audit flags are all candidate_setup_prediction metadata flags; their presence alone does not establish native observation bypass.

## Difficulty and next priorities

Combined normal/challenge success is 101/129 (78.29%) at L1, 84/198 (42.42%) at L2, and 80/258 (31.01%) at L3. The decline is consistent with accumulated state/evidence complexity, but these levels are different task sets and do not prove causality.

1. Make the generated-Python interface less brittle: support safe common syntax/helpers where appropriate, clearly separate query JSON from Python, and avoid forcing the model to transcribe evidence. Keep sandboxed read-only grounding and simulator-generated outputs; do not replace observations with native responses.
2. Reduce context-copying cost: provide bounded references/views or smaller evidence slices for generated programs; distinguish simulator computation limits from library copying overhead.
3. Improve verification around an explicit simulator-predicted state delta derived from successful write arguments and evidence. Compare semantic values and identities with expected state rather than assigning values from gold or inferring state from response text. Retain simulator scoring, including at N>1.
4. Record per-attempt errors, elapsed time, verifier result/error and bounded intermediate predicate results. Test scoring on both successful and unsuccessful completed candidates before another expensive N=16 sweep.

R29 removed one unnecessary host inspection gate, but prompt reminders did not prevent widespread syntax, context and semantic errors. A protocol/runtime and state-representation improvement is better supported by these findings than another prompt-only revision. This analysis does not change code or launch a new run.
