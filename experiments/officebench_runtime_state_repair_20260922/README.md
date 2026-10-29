# Runtime and simulator state repair (2026-09-22)

Comparison baseline: `dw-officebench-gpt54-contracts-subset-n124-20260922`.

- Materialize conversion input state and use native input-basename PDF output paths.
- Expose data-independent conversion, email and OCR semantics; validate stale missing-input and OCR string errors without leaking copied values.
- Reconstruct email messages from pristine MIME objects and ordered predicted sends, including both accounts, overwrites, body truncation and native directory order.
- Guard all-zero comparative selection with explicit per-check progress and stable ties. Per-check audits remain predictions, not ground truth.
- Flag DOCX masquerading as PDF in verification; leave native evaluator unchanged.
- Repair PRoot statx through ENOSYS fallback; provide LibreOffice configuration, synthetic NSS identity and explicit library/JRE paths without mounting the host /proc tree.
- Synthetic preflight validates Word/Excel PDF magic and extracted text, ls and network denial before the benchmark starts.

Validation: 172 tests passed, 6 skipped. Native synthetic conversion check passed without stderr. Azure runtime preflight also passed. The live GPT-5.4 private-input probe passed 4/4 checks, with 3 Python calls, zero Python failures and zero native candidate actions. End-to-end six-task subset completed: 93/99 exact selected observations; 3/6 native successes at each N; no exhausted candidates; zero simulator writes and 18 real commits. The final snapshot/recipient/order refinement is tested separately in `../officebench_final_state_subset_20260922/`.

The first Azure attempt failed before sampling because Java installation required `/usr/share/man/man1` in the minimal base image. The retry creates that directory; see `attempt1/` for the preserved startup failure. Active job: `dw-officebench-gpt54-runtime-state-r2-n124-20260922`.
