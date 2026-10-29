# OfficeBench reliability repair and targeted validation

Job: `dw-officebench-gpt54-reliability-v2-smoke-n124-20260921`.

The submitted job uses 12 tasks (indices 0, 2, 12, 20 in each of the three app-count subsets), with nested N=1,2,4 candidate pools: 36 task/N results and 48 unique candidates. It keeps GPT-5.4, temperature 1, max_steps 40, eight workers, and predict-all mode. `job.yml` pins the immutable Azure code asset; `source_manifest.json` records the uploaded source hashes.

Changes:

1. Read results identify their decoded representation and Python type. Simulation and local repair prompts explain native rendered XLSX, DOCX, PDF, OCR, ICS and text evidence. Safe regex match start/end/span methods are supported.
2. Local failed programs receive schemas alongside error and failed-code feedback. Exhausted persistent program errors stop identical full-rollout retries, retain counters, and are distinct from native action failures. Enabled diagnostic logs save the failing generated program and schemas.
3. The requested paradigm is preserved: every candidate observation is simulated against immutable evidence, candidates are selected, and only the selected trajectory is replayed in the environment. No unselected candidate receives a native score. Pure state models were additionally corrected for native Excel CLI conversion, None/no-op behavior, blank-cell and whitespace rendering, Word append/quote stripping, DOCX-versus-text format and read observation wrappers.
4. Partial-check coverage is separate metadata for selection, not success confidence. Unmet checks give zero whole-task confidence; unknown checks cap it below the success threshold. Whole-task voting honors the configured sample count. Removed the false claim that Word append requires reading first; uncertain predicted missing state is not automatically a deterministic rejection.
5. Restricted computation input/output budgets are both 2,000,000 characters with explicit failure, not truncation. LIST accepts file operands. READ on a directory returns a recoverable query error. Full predicted observations are retained for selected-commit comparison.

Local validation: 150 tests passed, 6 skipped in the OfficeBench, native differential, agent, remote-job and selection suites. Two additional targeted tests passed, verifying the five-vote per-check branch and file/directory query behavior. Native differential checks execute unchanged benchmark function bodies only on temporary local fixtures; they are not candidate rollouts.

Completed: all 36 task/N records are present. Successes increased from 2/12 to 3/12 at each N; exhausted errors fell to zero. At N=4, 11/12 pools still had all-zero confidence, so end-to-end reliability remains weak. See the final report at ../../officebench_reliability_final_20260921/report.md. `analyze.py` compares only matching previous task/N cells and separately measures exact and whitespace-trimmed selected-observation agreement. A targeted smoke run cannot establish full-benchmark reliability.
