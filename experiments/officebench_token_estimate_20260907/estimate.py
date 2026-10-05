"""Offline sensitivity estimate, NOT API usage measurement. Python stdlib only."""
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
# All sizes below are assumed tokens, except actual action strings (chars/4).
SCENARIOS = {
    'low': dict(solver_base=1200, observation=100, wm_input=1200, wm_output=100,
                verifier_base=1200, raw_context=500, retained_per_call=250),
    'base': dict(solver_base=2000, observation=300, wm_input=2500, wm_output=200,
                 verifier_base=2500, raw_context=2000, retained_per_call=1000),
    'high': dict(solver_base=3500, observation=1000, wm_input=5000, wm_output=400,
                 verifier_base=6000, raw_context=6000, retained_per_call=3000),
}


def actions(text):
    # Decode JSON to avoid counting pipes in shell commands/document text as steps.
    result, pos = [], 0
    while pos < len(text):
        marker = text.find('({"action":', pos)
        try:
            if marker < 0:
                raise ValueError('truncated action')
            start = marker + 1
            obj, size = json.JSONDecoder().raw_decode(text[start:])
            result.append(obj['action'])
            pos = start + size + 1
            if text[pos:pos+3] == ' | ':
                pos += 3
        except (ValueError, KeyError):
            end = text.find(' | ', pos)
            if end < 0:
                end = len(text)
            result.append({'raw': text[pos:end]})
            pos = end + 3
    return result


def stats(values):
    values = sorted(values)
    def quantile(q):
        at = (len(values)-1)*q
        lo = int(at)
        return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(at-lo)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                std=statistics.pstdev(values), p10=quantile(.1), p90=quantile(.9),
                total=sum(values), count=len(values))


def estimate(record, runs, p):
    solver_in = solver_out = 0
    trajectory_sizes = []
    for run in runs:
        history = 0
        for action in run:
            size = len(json.dumps(action, ensure_ascii=False))/4
            solver_in += p['solver_base'] + history
            solver_out += size
            # Approximate displayed argument truncation using 200-char per value cap.
            clipped = {k: str(v)[:200] for k,v in action.items()}
            history += len(json.dumps(clipped))/4 + p['observation'] + 12
        trajectory_sizes.append(history)
    if record['method'] == 'b2':
        return dict(input=solver_in, output=solver_out, total=solver_in+solver_out,
                    no_context_total=solver_in+solver_out, raw_only_total=solver_in+solver_out, wm_calls_proxy=0)
    breakdowns = record.get('candidate_verifier_percheck', [])
    atomic = sum((b or {}).get('total', 0) for b in breakdowns)
    whole = sum(b is not None for b in breakdowns)
    non_system = [sum(a.get('app') != 'system' and a.get('action') not in
                      ('switch_app','finish_task','got_stuck') for a in run) for run in runs]
    # Proxy, not measured: one prediction per non-system action plus one state and
    # one whole verdict per applicable rollout; clean reads can make this overestimate.
    # Python execution count supplies an observed floor/proxy for multi-turn work;
    # it is NOT an exact count of LLM calls. Atomic checks are separate direct calls.
    weights = [c + 1 + int(i < len(breakdowns) and breakdowns[i] is not None)
               for i,c in enumerate(non_system)]
    m = max(sum(weights), record.get('n_python_calls', 0))
    calls = [m*w/sum(weights) for w in weights]
    wm_in = m*p['wm_input']
    wm_out = m*p['wm_output']
    verifier_in = sum((b or {}).get('total', 0)*(p['verifier_base']+trajectory_sizes[i])
                      for i,b in enumerate(breakdowns))
    verifier_out = atomic*4
    # Metadata repeats across candidates; count each distinct selection event once.
    reranks = {json.dumps(x,sort_keys=True):x for x in record.get('candidate_comparative_rerank',[]) if x}
    for event in reranks.values():
        pairs = max(1, (event.get('response') or '').count('PAIR '))
        verifier_in += pairs*(p['verifier_base']+2*statistics.mean(trajectory_sizes))
        verifier_out += pairs*100
    inp = solver_in+wm_in+verifier_in
    out = solver_out+wm_out+verifier_out
    # Counterfactual: raw seed context on each WM call, and prior internal exchanges
    # retained across that rollout. Reset between candidates; same calls/outputs.
    extra = sum(c*p['raw_context'] + c*max(0,c-1)/2*p['retained_per_call'] for c in calls)
    return dict(input=inp, output=out, total=inp+out, no_context_total=inp+out+extra,
                raw_only_total=inp+out+m*p['raw_context'], wm_calls_proxy=m+atomic)


def main():
    rows, measured, hashes = [], {}, {}
    for method in ['ours','b2']:
        path = ROOT/f'qwen_offic_{method}/records.jsonl'
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        measured[method] = {}
        for n in [1,2,4,8,16]:
            cells = [r for r in records if r['n_rollouts']==n]
            decoded = [(r,[actions(a) for a in r['candidate_actions']]) for r in cells]
            assert len(cells)==300 and all(len(rr)==n for _,rr in decoded)
            measured[method][n] = dict(tasks=300, steps_per_candidate=stats([len(a) for _,rr in decoded for a in rr]),
                python_calls_per_task=stats([r['n_python_calls'] for r in cells]),
                zero_token_records=sum(r['tokens']==0 for r in cells))
            for r,runs in decoded:
                for scenario,p in SCENARIOS.items():
                    rows.append(dict(method=method,n=n,task_id=r['task_id'],subset=r['subset'],
                                     scenario=scenario,**estimate(r,runs,p)))
    with (OUT/'per_task.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for variant in ['b2','ours','ours_no_context','ours_raw_only']:
        for n in [1,2,4,8,16]:
            for scenario in SCENARIOS:
                rr=[r for r in rows if r['method']==('b2' if variant=='b2' else 'ours') and r['n']==n and r['scenario']==scenario]
                key={'ours_no_context':'no_context_total','ours_raw_only':'raw_only_total'}.get(variant,'total')
                summary.append(dict(variant=variant,n=n,scenario=scenario,**stats([r[key] for r in rr])))
    result=dict(assumptions=SCENARIOS,source_sha256=hashes,measured=measured,summary=summary)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for s in summary:
        if s['scenario']=='base':
            print(s['variant'],s['n'], 'mean/median/p90 K:',*[round(s[k]/1000,1) for k in ['mean','median','p90']])


if __name__=='__main__':
    main()
