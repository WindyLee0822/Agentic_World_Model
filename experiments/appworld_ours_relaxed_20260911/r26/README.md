# R26: Terminal read envelopes after inspection output

R24 payment task 024c982_2 passed native evaluation but received simulator score zero. Its verifier repeatedly generated inspection text followed by a JSON read-query envelope. The existing parser preserves that mixed output as facts and schedules no reads; this was reproduced in ../r24/verifier-parser-review.json. The truncated final verifier program prevents attribution of its exact score to this mechanism alone.

R26 relaxes the output format only for the opt-in AppWorld/Ours predictor. It accepts an explicit query envelope on the final stdout line after generated inspection text. The prefix and any explicit facts are retained together as evidence. It does not search intermediate or embedded text for requests. Malformed terminal requests still enter repair. The existing read-only endpoint gate and per-turn limits still control execution. Empty query lists preserve preceding inspection output.

Tests exercise actual generated Python through the read loop: the requested read executes, while a write request never reaches the native requester. Default Ours parsing stays unchanged. This change supplies requested evidence to the simulator; it does not calculate or override a verdict.

Only the opt-in subclass differs from R25. Submission requested for `dw-appworld-ours-relaxed-r26-n1-probe-20260913`, N=1 normal/L1 index34, one worker; service confirmation pending; live performance and the full BoN@16 goal remain unproven.

Completed diagnostic results (fetched 2026-09-14): [{"task_id": "024c982_2", "success": true, "candidate_scores": [0.0], "n_rollout_errors": 0, "n_commits": 1, "simulator_writes": 0, "candidate_errors": [null]}]. This N=1 diagnostic does not establish BoN@16 performance.
