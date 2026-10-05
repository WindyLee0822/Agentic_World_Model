# R30: AppWorld reliability repairs and DeepSeek N=1/2/4 sweep

**Current configuration:** six data shards sharing four candidates per task and reporting N=1,2,4. See [shared_pool/README.md](shared_pool/README.md). The 18 separate jobs described below were superseded at the user’s request.

R29 completed 585 tasks with 265 native successes. It aborted 146 candidates before replay and scored 199 successful candidates zero. Attempt traces contained 4,316 Python errors, including common safe syntax rejected by the runtime and library copying counted against the generated-program line limit.

## Implemented changes

The existing opt-in AppWorld/Ours predictor now enables a per-call Python compatibility mode. It supplies json and JSON-style true/false/null fallback bindings, accepts JSONDecodeError and type(value).__name__, permits assert and subscript/name deletion, and supports bounded signed-64-bit-sized integer literals. Failed assertions remain errors. The generated-code line limit counts only generated frames, so standard-library deepcopy internals do not consume it. Worker wall-clock and memory limits and the existing import/attribute/call restrictions remain. No global allowlists are mutated; default callers retain their old behavior even when sharing a worker with an experimental call.

The reference verifier now has two simulator-generated stages. First, a state-reconstruction program receives the task, successful writes, transitions, endpoint contracts and read-only grounding tools, but no evaluator checklist or gold. It produces structured changes, final_state and unresolved fields. Second, a verifier program compares this predicted state with the evaluator requirements and expected values. It can investigate missing evidence and emits per-check diagnostics before its verdict. The host checks the state envelope shape; it does not fill values, implement task-specific transitions, or override verdicts.

Verifier program results are logged with bounded output previews; each diagnostic event includes cumulative Python calls/failures so failed-attempt costs can be recovered. Existing full-response and program-error traces remain available. Input dictionaries are not dumped wholesale.

Every agent-facing API observation still goes through simulator prediction, including setup and errors. Candidate writes remain logical. Only the selected candidate is replayed natively. The changes affect the opt-in AppWorld/Ours subclass plus a default-disabled compatibility option in the shared Python runtime. The main repository implementation was not overwritten: submitted code is the isolated /tmp/appworld-r30-stage snapshot and durable overlays are in staged_changes/.

## Validation

77 distinct tests passed: 19 repair tests and 58 existing regressions. The 22 affected diagnostic/repair cases were rerun after adding cumulative diagnostics and passed. Tests exercise actual worker execution, compatibility isolation across consecutive calls, blocked filesystem/network/introspection/eval access, copy isolation, generated infinite-loop bounds, and gold-blind state reconstruction followed by correct/wrong-recipient/wrong-privacy verdicts. These tests establish the implementation behavior, not improved benchmark accuracy.

## Evaluation

18 jobs cover the Cartesian product of N={1,2,4}, split={test_normal,test_challenge}, and level={1,2,3}. Each uses DeepSeek-V4-Pro, ours, llm_wm, predict_all_actions, 40 agent steps, four workers, strict coverage and one task per child process. The manifest has exactly 585 unique task identities per N, or 1,755 decisions total. At most six jobs can occupy the configured compute cluster; actual concurrency also depends on other jobs and available capacity.

All jobs share one immutable registered code asset. Names follow dw-appworld-ours-r30-n{N}-{normal|challenge}-l{L}-20260915. See job-manifest.json for exact scope and submissions.json for accepted jobs. Submission is idempotent: existing names must reference the same code and verified command before reuse. Full performance and selection reliability remain to be measured from these jobs; no new accuracy claim is made.

## Submission result

All 18 jobs were accepted and their returned commands verified. Submission-time statuses were 16 Starting and 2 Queued; these are snapshots, not ongoing status claims. Each N covers all 585 tasks. See submissions.json and launch-summary.json. Benchmark accuracy remains pending.
