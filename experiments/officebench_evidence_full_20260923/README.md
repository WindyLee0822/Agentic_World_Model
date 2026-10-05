# Full OfficeBench evidence-r2 runs — 2026-09-23

Both jobs use the exact frozen production source validated by the eight-task evidence-r2 subset (183 tests passed, 6 skipped). No further code changes were made.

- GPT-5.4: `dw-officebench-gpt54-evidence-r2-full-n124-20260923`
- DeepSeek-V4-Pro: `dw-officebench-deepseek-evidence-r2-full-n124-20260923`

Each covers all 300 subtasks, levels 1–3, n=1,2,4 (900 expected decisions), four workers, 40 steps, temperature 1. Predict-all is explicit: predict every candidate observation, select, then commit only the selected trajectory. Native evaluator and dataset are unchanged. Strict coverage rejects incomplete runs.

The typed verifier, evidence routing, pure state transitions, email rendering, and selection repair are model-independent. DeepSeek is expected to benefit from these shared fixes, but its actual reliability has not yet been measured on this repaired version. The repaired verification and guarded/comparative selection settings are explicitly identical for both models, avoiding the older model-specific defaults. Unsupported checks retain five legacy verifier votes; supported typed checks calculate predicates on predicted state.

The source manifest records hashes of the frozen source. Job YAMLs and submission responses record exact settings and Azure job identities. Each job has a seven-day maximum runtime and a two-hour progress-stall timeout, matching the prior full-run allowance.

Submission confirmed: GPT-5.4 is Running; DeepSeek was accepted in Starting state. DeepSeek submission reused the exact registered GPT-5.4 code asset after the initial local-source submission stalled and was stopped before a job existed. Both accepted jobs have identical code asset IDs and full-dataset settings.
