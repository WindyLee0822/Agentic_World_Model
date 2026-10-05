# R29: full AppWorld evaluation at N=1

This run follows the September 14 request to summarize the submitted experiments, improve the current approach, and evaluate the whole dataset at N=1. It contains 585 tasks across test_normal/test_challenge and levels 1–3, with 16 workers and strict coverage. It is not a BoN@16 accuracy measurement.

## Experience from submitted runs

| Run | Scope | Result / lesson |
| --- | --- | --- |
| R16 | Full, N=2 | 367/585 (62.74%). Setup observations used native passthrough, so this does not satisfy the all-observations requirement. |
| R18 | Full, N=2 | 384/585 (65.64%). Setup metadata flags require care; they alone do not prove bypass. Neither historical run was N=16. |
| R19 | Relaxed full attempt | Failed under the run watchdog. Runtime management was a separate issue from prediction accuracy. |
| R20–R21 | Full attempts | Canceled after concrete exception-boundary defects were found. R21 also intercepted verifier failures that should have retained score-zero recovery. |
| R22 | Full, N=16, still running | Latest checkpoint: 149/585 completed, 87 successes. Extremely slow but progressed from 108 to 149. This partial set is not a final accuracy estimate. |
| R23 | File task, N=1 | Failed. Keeping final-turn evidence alone was insufficient. |
| R24 | File + payment, N=1 | 1/2 native successes; payment scored zero, file failed. Endpoint schema and rejection semantics were frequently invented. |
| R25 | Playlist task, N=1 | Failed. Read-error serialization instructions helped a local reproduction but did not establish live recovery. |
| R26 | Payment, N=1 | Native success, simulator score zero. Accepting terminal read requests after generated debug output fixed a protocol issue, but not verifier reasoning. |
| R27 | File + payment, N=1 | 2/2 native successes, both simulator scores 1. File: 325 selected Python calls, 27 errors, 2839.6 seconds. The payment verifier has a synthetic wrong-recipient false positive. |
| R28 | File + payment, N=1 | 2/2 native successes. File: score 1, 16 Python errors, 1624.2 seconds. Payment: score 0 despite native success, 4 verifier Python errors. Single-task timing comparisons are not broad performance estimates. |

## Changes from R28

- Allow a simulator-generated final verifier program to inspect fresh read evidence directly. Remove the host requirement for a separate facts-only round before accepting that program. Read-only grounding gates and Python-generated final output remain in force.
- Explain exactly where request arguments, read results, successful transitions, and computed facts live. In particular, generated facts remain under data['facts']; they are not new top-level keys.
- Put supported Python helper usage in the protocol before generation, addressing repeated __name__, JSONDecodeError, assert, and dir repair cycles.
- Tell the verifier to derive stored field representations from API semantics before comparison, and resolve candidate identities from actual arguments and read evidence rather than gold. These are instructions to the simulator, not host verdict overrides.

All changes are in the existing opt-in AppWorld/Ours subclass. Other source files are byte-identical to the R28 stage. Every solver-facing observation still uses simulator prediction, including setup and errors. Candidate writes remain logical only; native replay occurs only for the selected candidate. At N=1 the usual scoring/selection path has one candidate.

The N=1 full result and broad performance improvement are unmeasured until this new evaluation finishes. R22 is not canceled or restarted by this submission. See submission.json for the accepted Azure job, validation.json for tests, and isolation.json for source scope.
