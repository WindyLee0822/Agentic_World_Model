# R23 diagnostic: preserve final prediction evidence

R22's first all-candidate observation failure exposed a final-request omission: the read-loop budget-exhaustion prompt drops accumulated read summaries and computed facts. R23 retains the latest summaries/facts with existing redaction and length limits. Full values remain available to generated Python programs. This supplies evidence; it does not synthesize observations or verdicts.

Only the opt-in AppWorldOursModelDrivenPredictor enables the new path. Base and standard Ours final prompts remain byte-identical in tests. All other source is identical to R22 except the shared conditional and opt-in class attribute.

Validation: 3 new real read-loop tests and 26 existing prediction/routing regressions passed. The new tests cover final-turn reads/facts, default isolation, and absence of uninspected raw values in model text.

A separate N1 diagnostic on normal/L1 index9 (5a83b05_1) was submitted as `dw-appworld-ours-relaxed-r23-n1-probe-20260913` (code `620e6a50-6208-415e-bbf6-2025359dcfab/versions/1`), starting. It cannot establish BoN16 accuracy or be merged into R22's full585 results. R22 remains the active full benchmark, with its known evidence-prompt limitation recorded. No recovery or accuracy claim yet.


Diagnostic completed unsuccessfully: three observation failures, no native commit. See probe-analysis.json and final-records.jsonl. Evidence preservation alone is insufficient; do not promote as validated full setting.
