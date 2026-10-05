"""B1 sensitivity on ours' observed workloads; not an executed B1 benchmark."""
import csv
import json
from pathlib import Path
from estimate import ROOT, OUT, SCENARIOS, actions, estimate, stats

records = [json.loads(x) for x in (ROOT/'qwen_offic_ours/records.jsonl').read_text().splitlines()]
rows = []
for record in records:
    runs = [actions(a) for a in record['candidate_actions']]
    managed = estimate(record, runs, SCENARIOS['base'])
    checks = record.get('candidate_verifier_percheck', [])
    atomic = sum((b or {}).get('total', 0) for b in checks)
    # One call per non-system prediction, one state call per candidate,
    # one whole-verdict call where recorded; exclude adaptive read rounds/retries.
    direct = sum(sum(a.get('app') != 'system' and a.get('action') not in
                    ('switch_app','finish_task','got_stuck') for a in run)
                 + 1 + int(i < len(checks) and checks[i] is not None)
                 for i,run in enumerate(runs))
    managed_calls = managed['wm_calls_proxy'] - atomic
    for fixed in [1000, 4000, 8000, 15000]:
        # Retain earlier 2,500 local-input + 200 output budget, replacing adaptive
        # call proxy with direct calls. Add fixed data prefix to WM + atomic calls.
        total = managed['total'] - (managed_calls-direct)*2700 + fixed*(direct+atomic)
        rows.append(dict(n=record['n_rollouts'],task_id=record['task_id'],subset=record['subset'],
                         fixed_context_tokens=fixed,total=total,
                         fixed_context_calls=direct+atomic,
                         same_calls_total=managed['total']+fixed*managed['wm_calls_proxy']))
with (OUT/'b1_per_task.csv').open('w') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary = []
for n in [1,2,4,8,16]:
    for fixed in [1000,4000,8000,15000]:
        rr = [r for r in rows if r['n']==n and r['fixed_context_tokens']==fixed]
        assert len(rr)==300
        result = dict(n=n,fixed_context_tokens=fixed,**stats([r['total'] for r in rr]),
                      same_calls_mean=sum(r['same_calls_total'] for r in rr)/300)
        summary.append(result)
        print(n,fixed,'mean/median/p90 K',*[round(result[k]/1000,1) for k in ['mean','median','p90']])
(OUT/'b1_summary.json').write_text(json.dumps(dict(
    description='Counterfactual workload estimate on ours actions, base token assumptions; fixed prefix varies.',
    caveats='No B1 usage logs. Same solver trajectories, output sizes, state-call proxy and verifier workload. Direct calls assume no retries/adaptive rounds; same_calls_mean retains ours call proxy. Fixed prefix tokens are assumptions, not measured sizes.',
    summary=summary),indent=2)+'\n')
