# OfficeBench: explicit native semantics and current-state evidence

This change addresses the gap between executing a valid Python program and predicting a faithful native observation. Candidate actions are still simulated; the change does not execute unselected candidates.

## Implementation

- Added data-independent action contracts. Calendar reads specify ICS unfolding/text decoding, native timestamp conversion, event order, missing properties, headers, and empty/error responses. Excel and Word contracts specify native response templates and read wrappers.
- Enabled the restricted Python worker's `strftime` method. The worker previously allowed parsing with `strptime` but rejected the formatting operation used by the native calendar implementation.
- Predict-all Excel/Word edits receive typed `current_state` records computed from immutable evidence and supported pure transitions. Records distinguish exact current content/existence from unknown state. The contract explains relative resource keys and prevents treating dictionary membership as file existence. Unsupported transitions remain unknown; they are not silently interpreted as absence.
- Transition evidence preserves chronological order, includes multi-resource transitions once, and explicitly marks rejected writes as `applied=False`.
- Output validation routes invented write wording, contradictions of exact existence, raw calendar timestamps, and lossy clean Excel/Word reads through bounded local repair. Error feedback does not include the expected private content.
- Environment values are passed to generated Python as copied inputs, not interpolated into these action contracts. Regression tests place private sentinels in inputs and assert they are absent from model prompts, including repair prompts.

## Validation

The OfficeBench/native differential suite passed 126 tests with 6 skipped. The agent, AML-job configuration, and selection suites passed another 42 tests. The final focused regression rerun passed all 31 tests. The final GPT-5.4 check matched all four expected observations with 3 Python calls, 0 Python failures, and no candidate action execution. The generated programs used the supplied conversion and current-state contracts; the hidden event sentinel was absent from every prompt.

`live_check.py` runs GPT-5.4 against synthetic hidden operands with no native candidate-action interface. It checks calendar rendering, create/edit/read state consistency, and a genuinely missing file. `live_check.json` records exact-match results, generated programs, and repair counts. The private calendar value is asserted absent from every model prompt. This is a transition regression check, not a benchmark task-success estimate or a real-environment replay.

## Limits

The contracts and validators cover the actions described above; they are not a complete formal model of every OfficeBench app. Calendar format checks alone do not prove every event was copied correctly. Arbitrary shell transitions, document conversion, native-runtime failures, and verifier calibration remain outside this repair. No claim of full-benchmark reliability is made.
