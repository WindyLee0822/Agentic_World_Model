# Repaired OfficeBench full-dataset run

Job: `dw-officebench-gpt54-repaired-full-n124-20260922`. Final status: Completed.

[Azure ML job](https://ml.azure.com/runs/dw-officebench-gpt54-repaired-full-n124-20260922?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/external-collaborations/workspaces/M365Research-External-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47)

GPT-5.4, ours/llm_wm, all 300 subtasks (93/95/112 across levels 1/2/3), N=1,2,4: 900 task evaluations. Four workers, 40 steps, temperature 1. Predict all candidate observations, select the best trajectory, then commit only the selected trajectory. Native evaluation is unchanged.

Source is frozen from the validated runtime/state subset plus the final private-Python verifier repair, checked against its audited SHA-256. Runtime preflight is mandatory before sampling. Verifier uses per-check audits and five whole-task votes.

`source_manifest.json` records source hashes; `coverage.json` records expected coverage; `submitted_job.yml` pins the uploaded Azure code asset. `submission.json` and `verified_job.json` record the accepted remote job. Results and reliability audit: [analysis/report.md](analysis/report.md). All 900 records are present; native successes are 133/134/140 at N=1/2/4.
