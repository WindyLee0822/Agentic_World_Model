# R24: API contract instructions (diagnostic only)

R23's file probe failed all three attempts. Its final program in one attempt treated `api_documentation` as an endpoint list, although the context supplies a dictionary for the current endpoint. R22's payment-request failure also generated account-verification, friendship, and card prerequisites absent from the locally installed endpoint implementation. These findings motivate this prompt-only change; they do not prove every failure has the same cause.

R24 explains the exact documentation shape, use of the existing `data` variable, and the difference between a failure schema and a rejection condition. It asks the simulator to ground rejection conditions in the endpoint contract and state evidence. No outcome is hard-coded. R23's final-turn evidence preservation is retained.

Only `AppWorldOursModelDrivenPredictor` changes versus R23. Existing opt-in routing, generated setup and API observations, failure boundary, read tools, program execution restrictions, verifier, selection, and selected-candidate-only native replay remain in place.

The N=1 diagnostic covers normal/L1 indices 9 and 34 with two workers and the existing three-attempt policy. It is not a BoN@16 measurement or a substitute for R22's ongoing full benchmark. Neither recovery nor the 63% goal is established yet.

Validation: all 29 focused prediction, routing, and evidence regression cases passed (pytest exit 0). Source hashes confirm that only the opt-in write predictor differs from R23. AML submission has been requested; consult `submission.json` and `probe-status.json` for confirmed service status.

AML accepted the probe as `dw-appworld-ours-relaxed-r24-n1-probe-20260913`, code `e5afac10-cb0d-4bce-ad76-479fcf63841c/versions/1`; service status was Queued at 2026-09-13T17:07:17Z. The fetched command confirms N=1, task indices 9,34, normal split and two workers.

The local contract review in `api-contract-evidence.json` records source hashes and exact endpoint bodies without executing them. Venmo create_payment_request has receiver existence and self-request checks, but no account verification, friendship or payment-card gate. Spotify play_music with a song_id looks up the song and changes the queue, without a library-membership gate. These endpoint sources contain behavior not fully specified by the short docstrings; R24 only clarifies how the existing documentation is used and does not add source evidence. The review is not a worker-byte comparison or proof of the causes of all native failures.

Completed diagnostic results (fetched 2026-09-14): [{"task_id": "024c982_2", "success": true, "candidate_scores": [0.0], "n_rollout_errors": 0, "n_commits": 1, "simulator_writes": 0, "candidate_errors": [null]}, {"task_id": "5a83b05_1", "success": false, "candidate_scores": [0.0], "n_rollout_errors": 1, "n_commits": 0, "simulator_writes": 0, "candidate_errors": ["WM rollout seed 9000 failed after 3 attempt(s): SimulatorObservationFailure: Simulator observation unavailable"]}]. This N=1 diagnostic does not establish BoN@16 performance.
