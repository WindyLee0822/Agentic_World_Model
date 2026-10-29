# R30 shared candidate pool — current sweep

This supersedes the 18 separate N jobs. Six jobs partition the data by normal/challenge and levels 1/2/3, covering 585 unique tasks. Each runs DeepSeek-V4-Pro, ours, llm_wm, predict_all_actions, with --n "4,2,1" and four workers.

The runner generates four candidate slots first, then reuses their fixed first-two and first-one prefixes. Selection uses simulator scores for each prefix; only the selected trajectory for each reported N is replayed in a fresh native environment for evaluation. Native outcomes do not enter candidate generation or selection. This yields 1,755 result records from 2,340 candidate slots before retries, instead of 4,095 slots for independent N sweeps.

The only additional source change versus tested R30 is an opt-in harness cache rule: after retry exhaustion, failed candidates are cached too, so smaller prefixes cannot regenerate them. Activation requires AppWorld, ours, predict_all_actions, and TTS_APPWORLD_FIXED_CANDIDATE_POOL=1. Four tests verify fixed-prefix reuse including failures and unchanged behavior for default callers, OfficeBench, and B1. All original R30 observation/runtime/verifier repairs remain.

See job-manifest.json for scope and submissions.json for accepted jobs. Earlier jobs are retained as superseded history in the parent directory; their outputs must not be mixed into this sweep. Accuracy remains pending.
