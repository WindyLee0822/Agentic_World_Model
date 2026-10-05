# R25: Read-error serialization instructions (not yet benchmarked)

R22 task 634f342_2 failed with 15/16 candidate errors. Several complete short traces request file_system.show_file with `path` although the native endpoint requires `file_path`; they then print the grounding error string directly or parse it using json.loads.

The reproduction in ../r22/failures/634f342_2/read-error-reproduction.json uses native Requester argument validation and the actual simulator Python worker without executing an endpoint. Native validation produces a plain-text error; raw print fails JSON normalization, json.loads fails, and generated print_json preserves the value as valid JSON. The original trace does not contain execution causes, so this verifies the mechanism rather than proving every candidate's cause.

R25 appends serialization guidance only to clean/dirty read prompts in the existing opt-in AppWorld/Ours subclass. It tells the simulator to use json_value for conditional decoding and print_json for its final prediction, including strings. It does not change validation, fabricate observations, execute candidate writes, or alter verifier/selection. R24 write instructions and R23 final-turn evidence remain inherited. Every other Python source file is byte-identical to R24.

Submission requested for `dw-appworld-ours-relaxed-r25-n1-probe-20260913`: N=1, normal/L2 index 7 (634f342_2), one worker; service confirmation pending. See submission.json when available. All 29 focused regression tests passed; the full BoN@16 target remains unproven. R22 and R24 continue independently.

Completed diagnostic results (fetched 2026-09-14): [{"task_id": "634f342_2", "success": false, "candidate_scores": [0.0], "n_rollout_errors": 1, "n_commits": 0, "simulator_writes": 0, "candidate_errors": ["WM rollout seed 7000 failed after 3 attempt(s): SimulatorObservationFailure: Simulator observation unavailable"]}]. This N=1 diagnostic does not establish BoN@16 performance.
