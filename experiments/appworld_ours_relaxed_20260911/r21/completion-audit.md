# R21 completion audit: incomplete

- Relax R16 constraints: inherited model-driven write success/rejection prediction, disabled deterministic completion deferral and answer-key rejection. Simulator remains reference-assisted, program-only, and read-grounded; R21 is not an unconstrained setting. See R20 implementation and R21 stage.json.
- Every API observation predicted: R20 response-routing audit remains applicable with its operational-error gap addressed by the new host boundary. Native tests prove withholding caught/uncaught internal errors and preserving predicted rejections; the plain model-error case also passed. Remaining: full runtime evidence and review of routes outside the requester.
- Simulator has information: full API docs, schemas, arguments, frozen time, history, logical writes, and read-only evidence tools are inherited unchanged. Exactness of all generated observations is not established by availability of inputs.
- Simulator selection: scoring and selection code unchanged from R20 (see ../r20/selection-routing-audit.md). No native evaluation of unselected candidates; full run records still pending.
- Other datasets/methods unaffected: R21 isolation.json verifies every Python source identical to R20 except the boundary module and its installation in the existing AppWorld/Ours/predict-all/model-driven branch. This is source isolation, not empirical performance measurement across datasets.
- BoN16 around 63%: job dw-appworld-ours-relaxed-r21-n16-full-20260913 queued. Exact585-task benchmark pending; do not estimate from partial or R20 results.
- Validation: 10 focused boundary tests, 4 native execution cases, 26 staged prediction/routing regressions passed. Green checkpoint metadata alone cannot establish observation fidelity.

R20 remains live as separate measurement evidence but has a known observation-route defect. Do not merge its records into R21. R16/R18 historical full runs were N2, not N16.


Superseded: confirmed verifier exception-wrapping regression; canceled. R22 is the active experiment. R21 cannot establish goal completion.
