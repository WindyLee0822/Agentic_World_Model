# AppWorld Ours performance repair

**Current full R16 result: 367/585 official BoN@2 successes (62.74%).** All 585 expected
identities are present exactly once. See [R16_RESULT.md](R16_RESULT.md) for the current result
and remaining contract issue. The full goal remains open because existing exact setup API
responses do not prove literal simulator prediction of every observation.

The following sections retain historical experiment notes; their partial-run statements
are superseded by the current full R16 record audit.

The goal is **overall BoN@2 >60% on all 585 tasks**, with simulator-generated
observations and simulator selection. It is **not yet achieved**. At least 352
successes are required; a pilot cannot establish completion.

## Earlier R14 full run

`dw-appworld-ours-verified-r14-full-20260907` is running on AML. The code is
frozen while it accumulates benchmark evidence. It uses DeepSeek-V4-Pro and pinned
AppWorld 0.2.0.dev0 commit `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a` / data 0.2.0,
N=2, all 585 tasks, 40 agent steps and temperature 1. Configuration:
`aml/job.appworld.ours.verified-r14-full.yml`.

The exact source snapshot is `/tmp/appworld-full-r14-submit`; hashes are in
`full-r14.json`. The worktree now matches R16; these hashes retain R14 provenance. R11 was
canceled after its verifier schema failures and false-positive selection were
reproduced and repaired. R8 was canceled after private funding reads were found
unavailable. R2/R3 submissions never registered because SDK Git metadata
collection hung in the damaged checkout. Subsequent submissions run from clean
staging directories; Git history and the incomplete index were not overwritten.

## Measured results

| Evaluation | Official BoN@2 successes | Scope |
|---|---:|---|
| Original `dw-appworld-ours-llm-wm-deepseek-v4-pro-n12-20260904` | 7/585 | Full target benchmark |
| First repaired AML pilot | 2/6 | One task per split/level |
| R4 AML pilot | 3/6 | Same six tasks, completed |
| Local R6 `b3bdcc1_1` | 1/1 | Target-version N=2 functional task check |
| R7 AML pilot | 2/6 | Same six tasks |
| R11 full run | Superseded | Partial diagnostic results |
| Local R14 regression tasks | 2/2 | Two target-version N=2 task checks |
| R14 full run | Running; see `partial-r14-gate.json` | All 585 tasks; coverage incomplete |

R4 succeeded on `fd1f8fa_1`, `2d9f728_1`, and `23d431c_1`; it failed on
`3d9a636_1`, `5238afc_1`, and `b3bdcc1_1`. R7 exposed a false-positive selection:
both Spotify candidates received SUCCESS even though one fetched only the first
page of liked songs. The inherited verifier explicitly permits UNKNOWN checks
to pass; the working-tree verifier now requests grounding and requires proof.

Earlier local `024c982_1` smoke successes used AppWorld 0.1.3.post1/data 0.1.0.
They are not matched target-benchmark improvement evidence. `validation.json`
marks this version mismatch.

## Repairs included in the full snapshot

- An Ours-only subclass fixes the `predicted_write`/`predict_write` callback
  mismatch. Baseline predictors retain their original behavior.
- Model transport runs in a separate host process initialized before AppWorld's
  SafetyGuard. AML OBO token refresh previously failed because the guard blocked
  `urllib.request.urlopen`. Solver and computation-worker restrictions remain
  intact. Interrupted waits discard stale response IDs.
- Python output limits accommodate native pages larger than 20K characters,
  while retaining finite limits and the same restricted AST, filesystem and
  network boundaries. Generated fact names survive successive inspection rounds.
- The simulator receives exact read API names and parameter names. Program token
  floors and ten inspection rounds allow grounded programs to finish.
- Write predictions receive ordered successful transitions, with explicit
  pristine-state and schema-inspection guidance. Hand-built Venmo funding
  responses and synthetic IDs are disabled for Ours.
- Malformed generated responses cannot fall through to inherited synthetic
  observations. Budget exhaustion invalidates candidates.
- Nine audited GET download endpoints are treated as simulated writes because
  they create files and update counters. They cannot be grounding tools.
- An isolated requester view enables only the audited private
  `admin.show_card_balance` read, with parameter validation. The solver's
  requester is unchanged. A native probe confirms both properties.
- Ours budgets start after setup, use 1800/2400-second rollout defaults and
  600-second cells, and retain explicit user budget overrides. Agent history
  records the code actually executed when a response contains multiple blocks.

## Additional repairs included in R14

Affected write-precondition reads can use nested simulator predictions, with
reuse of generated observations only for matching arguments, credentials and
write-history version. New writes invalidate reuse. Generated inspection stdout
can be retained verbatim as facts; embedded query examples are not executed.
Type-inspection errors suggest safe repairs without weakening the Python guard.
A stricter reference verifier gathers missing read-only evidence and no longer
accepts UNKNOWN substantive checks as success. The repaired verifier returns FAIL for the captured incomplete Spotify candidate
and SUCCESS for the complete candidate, without executing either candidate's
writes. Local N=2 runs now pass `cef9191_1` and `21abae1_3`, which failed in R11.

## Evaluation contract and validation

Agent reference answers, instruction preflights and deterministic clean-read
shortcuts are disabled. No candidate writes execute for selection; only the
selected candidate is replayed for official evaluation. The record audit in
`aml/src/check_appworld_ours_goal.py` requires all 585 unique expected identities,
352 successes, simulator selection, no candidate grounded scores and at most one
commit. Record flags supplement source/runtime inspection; they do not replace it.

The R11 full snapshot passed 30 targeted predictor/transport/coverage tests and a
native private-read probe. Earlier focused AppWorld/selection checks passed 151
cases. The expanded grounding suite's nine callback failures also reproduce
against the saved baseline. Earlier isolation checks had 152 passed, 8 skipped,
and four pre-existing failures (one callback typo and missing tau2 dependencies).
See adjacent logs and manifests for revision-specific evidence.

All repairs are selected only for AppWorld `method=ours`. Shared hooks preserve
baseline defaults. No other dataset implementation or shared model client was
changed.


## R14 verification and remaining work

The focused AppWorld/selection suite passes 162 tests. New regressions cover
inspection before verdicts and retention of generated candidates when verifier
programs fail. Failed verification gets zero confidence and is recorded in
metadata; it no longer triggers regeneration of an entire completed candidate.
Ours-only solver guidance requests complete pagination and the requested answer
format. The verifier's copied input types are explicit, and generated programs
must inspect new read values before judging them.

Earlier stricter-verifier probes rejected both fixtures or failed to execute.
Those failures exposed unsupported Python syntax, wrong read shapes and stale
playback flags. `probe_verifier_selection_v5.log` records the successful repaired
comparison. The native private-read probe and baseline-isolation evidence remain
applicable. No full >60% result is verified yet.

Latest live coverage and contract results are in `partial-r14-gate.json`. The frozen
R14 run remains active; its frozen source is tracked separately.

## Separate R15 experiment (not in this worktree or the full run)

R14 local N=2 diagnostics failed `3d9a636_1` and `5238afc_1`. The friend-sync
agent included all phone contacts instead of the relationship-filtered friends.
The Amazon agent's product-name substring excluded a valid category member and
it supplied an unrequested answer. A successful retry with another card also
left the earlier checkout rejection unresolved in simulator bookkeeping.

A separate source snapshot at `/tmp/appworld-r15-experiment` is testing generic
structured-category/relationship guidance, answer omission for action-only tasks,
collection reconstruction before pagination, and Ours-only checkout intent matching
across funding cards. At that stage, the worktree and AML R14 source remained unchanged.
The checkout-intent check passes, including distinct-address and baseline checks;
matched target-version task evaluations and the pagination comparison are running.

The R14 pagination probe returned page sizes `[27, 0, 0]` at a requested limit of
20. Its initial set-consistency checker overlooked the page-size bound; the checker
now includes that bound. The captured outputs demonstrate the failure independently
of the earlier overly permissive Boolean result. This diagnostic executes no native
candidate writes and is not an official task score.

The matched pagination comparison is now complete (`r14-pagination-v2.log`,
`r15-pagination-v2.log`). Both fail: R14 produces `[20, 17, 0]` for 27 unique
records (ten duplicates); R15 produces `[20, 0, <dict>]`. The pagination prompt
is not validated and has not been adopted into the worktree. The first probe
also omitted predictor login history, which broke automatic token hydration;
v2 fixes that setup and explicitly enables predict-all program-only mode.

Latest completed local evidence is recorded in `latest-local-regressions.json`:
R14 passes the previously failing Venmo funding task `b3bdcc1_1`; R15 passes
friend-sync `3d9a636_1`; R16 passes note-counting `7847649_3`. Each uses N=2,
simulator selection, no candidate grounded score, and one selected real replay.
R16 excludes the unvalidated R15 pagination prompt and adds guidance to exclude
headings, legends and examples when counting actual item lines. The native note
had six completed activities plus an `[x]` marker in its explanatory heading;
R14's two candidates counted seven. R15's shopping regression is still running.
These local results do not establish the full goal or validate a full R16 run.
At that stage, the root source remained R14.

R15's shopping regression also passed official N=2 evaluation. R16 now rechecks
both stateful cases with the unvalidated pagination prompt removed. Its focused
suite passes 165 checks: 163 passed initially; two required the unchanged experiment
runner omitted from the local test copy, and both passed after that fixture was added.
The initial remote-filesystem test run was stopped after stack inspection showed a
filesystem wait. Source behavior was not changed to resolve either test-run issue.

## Parallel R16 full evaluation

`dw-appworld-ours-combined-r16-full-20260907` has registered on AML. It evaluates
all 585 tasks at N=2 with 16 workers, using the same model, target AppWorld/data
versions and simulation/selection settings. R14 continues independently. The
cluster is configured for up to six nodes; actual allocation is determined by AML.

R16's frozen source is `/tmp/appworld-full-r16-submit`, with hashes in
`full-r16.json`. It matches the validated local experiment: four official N=2
regressions pass and 165 focused checks pass. The root worktree now matches the submitted R16 source while
both results accumulate. Their records must remain separate, and any final delivered
source must match the full run used to establish the goal. Neither run yet has
full coverage or a verified >60% full-benchmark result.

## Recovery after the local runtime reset

At 2026-09-07 15:39 UTC, both AML full evaluations were confirmed Running.
R14 had 203 successes from 295 completed tasks; R16 had 322 successes from 470.
Their record audits reported no contract violations. R16 still needed 30 successes
and all remaining 115 task decisions before the full goal could be verified.

Temporary local probe sessions and `/tmp` snapshots were lost. The exact R16 code
asset was recovered to `snapshots/r16` and its five source hashes verified;
results and recovery helpers now live
in this report directory. `checkpoint.json` records the current recovery state.
The R17 filesystem/credential experiment was never adopted into either full run.
No AML evaluation was restarted because of the local runtime reset.

## R16 worktree promotion

The two R16 source differences and checkout-retry regression test are now in the
worktree. All five repaired source hashes match `full-r16.json`; syntax checks pass.
The changes add Ours-only category/count/answer guidance and clear a declined
checkout when the same purchase succeeds with another card. Distinct shipping
addresses retain separate intent, and baseline checkout behavior is unchanged.
At promotion, R16 had 327 official successes from 475 completed tasks. All 585
decisions and at least 352 successes remain required.

Routing preserves the existing exact authentication, control, and API-documentation
setup responses. Substantive task observations are simulator-generated under
`predict_all_actions`; no additional exact-read bypass was introduced by these repairs.
