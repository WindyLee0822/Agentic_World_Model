# OfficeBench evidence repair, 2026-09-23

This revision preserves prediction of all candidate observations, selection using predicted outcomes, and native execution of only the selected trajectory. The native evaluator and task fixtures are unchanged. Pure helpers process copied inputs without inserting their contents into model prompts.

## Changes

- Route email checks to `emails/{username}` and infer document type for existence-only checks. Always build mailbox evidence, including untouched mailboxes.
- Apply supported native predicates directly to predicted state: existence, keyword containment and whole-predicate negation, dictionary-shaped cell checks, paired-file diffs, and conservative exact matching. Unsupported operators retain the previous verifier. Missing evidence remains UNKNOWN. Numeric spreadsheet types lost during decoding remain UNKNOWN.
- Materialize supported gold-backed state edits instead of supplying only an initial-state/write-log description.
- Render single-email reads with native headers, full body, and suffix; maintain copied state for bounded move/copy and directory creation commands.
- Keep the incumbent eligible in an all-zero candidate pool; uncertain checks alone cannot justify switching.
- Keep repeated-action feedback in agent history, outside predicted application observations; skip malformed actions during commit replay.
- Expose only the presence of the native LLM credential file as a runtime capability, allowing prediction of its missing-file error without reading credentials.

## Evaluation design

Job: `dw-officebench-gpt54-evidence-r2-subset-n124-20260923`.

Eight targeted tasks × n=1,2,4 = 24 records, GPT-5.4, temperature 1, four workers, 40 steps. The baseline consists of the matching records in `dw-officebench-gpt54-repaired-full-n124-20260922`.

Tasks: 1-3/0 (move), 1-4/0 (cell checks), 1-7/0 (diff), 1-8/0 (empty spreadsheet exact match), 1-20/0 (email read/negation), 3-12/0 (mailbox evidence), 3-65/0 (document existence/email/selection), 3-82/0 (incompatible decoder negative control).

This is a targeted regression subset, not a random sample. Different sampled trajectories can change the mixture of observed actions. Exact observation agreement is measured only on committed trajectories: unselected candidates have no native oracle. One run per setting cannot establish statistical significance.

Known environment/fixture controls: 1-7/0 references an unprovisioned cache input; 3-82/0 asks the evaluator to decode a DOCX filename with the XLSX decoder. The simulator should recognize failure without changing these native conditions.

## Remaining scope

General shell effects, dirty directory listings, calendar reconstruction, and some document conversions still use fallible prediction paths. The bounded pure helpers are not a complete OfficeBench emulator. Typed checks improve verdict computation, but cannot make incorrect predicted state correct.
