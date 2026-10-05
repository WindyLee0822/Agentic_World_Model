# R16 full benchmark result

R16 records contain **367 official successes / 585 tasks (62.74%)** at BoN@2,
compared with **7/585 (1.20%)** for the original run: an increase of 61.54 percentage points.
The full record gate verifies all expected task identities exactly once, no missing or extra tasks,
and no recorded contract violations. See `full-r16-gate.json` and `r16-completion-audit.json`.

Job: `dw-appworld-ours-combined-r16-full-20260907`.
Configuration: `aml/job.appworld.ours.combined-r16-full.yml`.
The five repaired runtime source hashes match the submitted R16 snapshot (`full-r16.json`).
The selection source hashes still match the inspected simulator-selection implementation.

The **whole goal is not yet complete**. Substantive task observations are simulator-generated,
and selection uses simulator predictions before the selected candidate's official replay.
Existing supervisor, API-documentation and login responses remain exact. Whether these setup
responses are within the requested every-observation scope remains unresolved; record flags
cannot establish that literal requirement. The simulator verifier also retains the existing
reference checklist/gold-value input, as documented in `selection-source-audit.json`.

The repairs are scoped to AppWorld Ours, with baseline defaults retained in shared hooks.
Persisted validation includes 165 focused R16 checks, four target-version official N=2 regression
successes, and the earlier baseline isolation comparison. These checks do not constitute a new
full evaluation of every other method/dataset. See the historical README and adjacent logs.
